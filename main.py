import random
import typing
import copy
from collections import deque
delta = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0)
}
# --- GLOBALS für Zobrist-Hashing ---
ZOB_SNAKE = {}   # wird in start() pro Snake-ID initialisiert
ZOB_FOOD  = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
current_hash = 0  # globaler 64-Bit Hash
# === INFO ===
def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Henrik-Team",
        "color": "#736CCB",
        "head": "beluga",
        "tail": "curled"
    }
transposition_table = {}
##hash für ganze Board 
def init_hash(state):
    """Initialer Zobrist-Hash für das ganze Board."""
    h = 0
    for snake in state['board']['snakes']:
        sid = snake['id']
        # 2D-Array für diese Snake anlegen, falls noch nicht geschehen
        if sid not in ZOB_SNAKE:
            ZOB_SNAKE[sid] = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
        for seg in snake['body']:
            h ^= ZOB_SNAKE[sid][seg['x']][seg['y']]
    for f in state['board']['food']:
        h ^= ZOB_FOOD[f['x']][f['y']]
    return h

# === GAME START ===
def start(game_state: typing.Dict):
    """Spielstart: Zobrist-Tables initialisieren, TT leeren und initialen Hash setzen."""
    global current_hash, transposition_table
    transposition_table.clear()    # Alte Einträge entfernen
    # Stelle sicher, dass ZOB_SNAKE für alle IDs da ist
    for snake in game_state['board']['snakes']:
        sid = snake['id']
        if sid not in ZOB_SNAKE:
            ZOB_SNAKE[sid] = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
    current_hash = init_hash(game_state)
    print("GAME START")
#----------------Funktionen------------------------------

def get_enemy_id(game_state: typing.Dict) -> str:
    """Gibt die ID des Gegners zurück."""
    return next(
        s['id']
        for s in game_state['board']['snakes']
        if s['id'] != game_state['you']['id']
    )

def apply_moves(game_state: typing.Dict, move_dict: typing.Dict[str,str]) -> typing.List[tuple]:
    """Wie vorher, plus Zobrist-XOR-Updates."""
    global current_hash
    board = game_state['board']
    food_set = {(f['x'], f['y']) for f in board['food']}
    changes = []

    for snake in board['snakes']:
        sid = snake['id']
        mv = move_dict[sid]
        dx, dy = delta[mv]
        head = snake['body'][0]
        new_head = {"x": head["x"] + dx, "y": head["y"] + dy}

        # --- ZOBRIST: alten Kopf entfernen, neuen hinzufügen ---
        current_hash ^= ZOB_SNAKE[sid][head['x']][head['y']]
        current_hash ^= ZOB_SNAKE[sid][new_head['x']][new_head['y']]

        snake['body'].insert(0, new_head)

        if (new_head["x"], new_head["y"]) in food_set:
            # Food gefressen: Food-Zobrist entfernen
            current_hash ^= ZOB_FOOD[new_head['x']][new_head['y']]
            for i,f in enumerate(board['food']):
                if (f['x'], f['y']) == (new_head['x'], new_head['y']):
                    removed = board['food'].pop(i)
                    changes.append((sid, True, removed))
                    break
        else:
            # Schwanzsegment raus – Zobrist ebenfalls rückgängig machen
            tail = snake['body'].pop()
            current_hash ^= ZOB_SNAKE[sid][tail['x']][tail['y']]
            changes.append((sid, False, tail))

    return changes

def undo_moves(game_state: typing.Dict, changes: typing.List[tuple]):
    """Undo plus Zobrist-XOR zurückdrehen."""
    global current_hash
    board = game_state['board']

    for sid, ate, seg in reversed(changes):
        snake = next(s for s in board['snakes'] if s['id'] == sid)
        head = snake['body'][0]
        # Kopf entfernen (neuester)
        snake['body'].pop(0)
        current_hash ^= ZOB_SNAKE[sid][head['x']][head['y']]

        if ate:
            # Food wieder rein – Hash erneut XORen
            board['food'].append(seg)
            current_hash ^= ZOB_FOOD[seg['x']][seg['y']]
        else:
            # Schwanz wieder anhängen und Hash
            snake['body'].append(seg)
            current_hash ^= ZOB_SNAKE[sid][seg['x']][seg['y']]

def evaluate_move_3ply(start_move: str,
                       game_state: typing.Dict,
                       is_move_safe: typing.Dict[str, bool],
                       evaluation_function: typing.Callable,
                       alpha: float,
                       beta: float,
                       depth: int) -> float:
    """
    3-Ply Lookahead mit korrektem Zobrist-Key nach apply_moves
    und frischem is_move_safe auf jeder Ebene.
    """
    global current_hash
    alpha_orig, beta_orig = alpha, beta
    me = game_state['you']['id']
    enemy_id = get_enemy_id(game_state)

    # 1) Ersten Zug anwenden
    changes1 = apply_moves(game_state, {me: start_move})

    # 2) Jetzt TT-Lookup auf Basis des aktualisierten Hash
    key = (current_hash, depth)
    cached = lookup_in_tt(key, depth, alpha, beta)
    if cached is not None:
        undo_moves(game_state, changes1)
        return cached

    # 3) Restlicher 2-Ply-Search: Gegnerzug + unser zweiter Zug
    best_score = -float('inf')
    for em in delta:
        # sichere Gegnerzüge neu berechnen
        safe_e = {m: True for m in delta}
        avoid_collisions(
            game_state['board']['snakes'][1]['body'][0],
            game_state['board']['snakes'][1]['body'],
            game_state['board']['snakes'],
            safe_e,
            game_state['board']['width'],
            game_state['board']['height'],
            enemy_id
        )
        if not safe_e.get(em):
            continue

        changes2 = apply_moves(game_state, {enemy_id: em})
        for om in delta:
            # sichere eigene Züge neu berechnen
            safe_o = {m: True for m in delta}
            avoid_collisions(
                game_state['you']['body'][0],
                game_state['you']['body'],
                game_state['board']['snakes'],
                safe_o,
                game_state['board']['width'],
                game_state['board']['height'],
                me
            )
            if not safe_o.get(om):
                continue

            changes3 = apply_moves(game_state, {me: om})
            score = evaluation_function(om, game_state, safe_o)
            undo_moves(game_state, changes3)

            if score > best_score:
                best_score = score
                alpha = max(alpha, score)
            if alpha >= beta:
                undo_moves(game_state, changes2)
                undo_moves(game_state, changes1)
                store_in_tt(key, depth, best_score, 'LOWERBOUND')
                return best_score

        undo_moves(game_state, changes2)

    # 4) TT-Eintrag speichern
    if best_score <= alpha_orig:
        etype = 'UPPERBOUND'
    elif best_score >= beta_orig:
        etype = 'LOWERBOUND'
    else:
        etype = 'EXACT'
    store_in_tt(key, depth, best_score, etype)

    # 5) Ersten Zug rückgängig machen und Ergebnis liefern
    undo_moves(game_state, changes1)
    return best_score
           
def lookup_in_tt(hash_key, depth, alpha, beta):
    entry = transposition_table.get(hash_key)
    if not entry or entry['depth'] < depth:
        return None

    val = entry['value']
    etype = entry['type']  # 'EXACT', 'LOWERBOUND', 'UPPERBOUND'

    # Bei Exact-Einträgen sofort zurückliefern
    if etype == 'EXACT':
        return val

    # Bei Lower-Bound: wir wissen score ≥ val
    if etype == 'LOWERBOUND' and val > alpha:
        alpha = val

    # Bei Upper-Bound: wir wissen score ≤ val
    if etype == 'UPPERBOUND' and val < beta:
        beta = val

    # Wenn die Bounds sich überlappen, können wir prunen
    if alpha >= beta:
        return val

    return None

def store_in_tt(hash_key, depth, value, entry_type):
    transposition_table[hash_key] = {
        'depth': depth,
        'value': value,
        'type': entry_type  # 'EXACT', 'LOWER', 'UPPER'
    }

def determine_mode(my_length, enemy_length, my_health):
    if my_health < 25:
        return "emergency"
    elif my_health < 40 or my_length <= enemy_length + 1:
        return "recovery"
    elif my_length >= enemy_length + 5:
        return "kill_mode"
    elif my_length >= enemy_length + 2:
        return "aggressive"
    else:
        return "normal"

def calculate_free_space(my_head, game_state, max_limit=50):
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    occupied = {(segment['x'], segment['y']) for snake in game_state['board']['snakes'] for segment in snake['body']}
    queue = deque([(my_head['x'], my_head['y'])])
    visited = {(my_head['x'], my_head['y'])}
    free_space, quality_score = 0, 0
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        free_space += 1
        free_neighbors = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_width and 0 <= ny < board_height and (nx, ny) not in occupied:
                free_neighbors += 1
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        quality_score += free_neighbors
        if free_space >= max_limit:
            break
    return free_space, quality_score

def calculate_enemy_free_space(enemy_head, game_state, max_limit=50):
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    occupied = {(segment['x'], segment['y']) for snake in game_state['board']['snakes'] for segment in snake['body']}
    queue = deque([(enemy_head['x'], enemy_head['y'])])
    visited = {(enemy_head['x'], enemy_head['y'])}
    free_space = 0
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        free_space += 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_width and 0 <= ny < board_height and (nx, ny) not in occupied and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
        if free_space >= max_limit:
            break
    return free_space

def is_true_head_on_risky(move, my_head, my_length, game_state):
    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    occupied = {(segment['x'], segment['y']) for snake in game_state['board']['snakes'] for segment in snake['body']}

    for snake in game_state['board']['snakes']:
        if snake["id"] == game_state["you"]["id"]:
            continue
        enemy_head = snake["body"][0]
        enemy_length = snake["length"]
        enemy_health = snake["health"]
        enemy_moves = []

        for ex, ey in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = enemy_head["x"] + ex, enemy_head["y"] + ey
            if 0 <= nx < board_width and 0 <= ny < board_height and (nx, ny) not in occupied:
                enemy_moves.append((nx, ny))

        if (new_head["x"], new_head["y"]) in enemy_moves:
            if enemy_length > my_length:
                return "death"
            elif enemy_length == my_length:
                # Hier Vermeidung auch bei Gleichstand!
                return "death"
            else:
                return "advantage"

    return "safe"

def avoid_collisions(my_head, my_body, snakes, is_move_safe, board_width, board_height, my_id):
    # Wand-Kollision
    for move, (dx, dy) in delta.items():
        nx, ny = my_head["x"] + dx, my_head["y"] + dy
        if not (0 <= nx < board_width and 0 <= ny < board_height):
            is_move_safe[move] = False

    # Eigener Körper – ohne den Schwanz (letztes Segment)
    for segment in my_body[1:-1]:
        for move, (dx, dy) in delta.items():
            if segment["x"] == my_head["x"] + dx and segment["y"] == my_head["y"] + dy:
                is_move_safe[move] = False

    # Gegnerkörper
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        for segment in snake["body"]:
            for move, (dx, dy) in delta.items():
                if segment["x"] == my_head["x"] + dx and segment["y"] == my_head["y"] + dy:
                    is_move_safe[move] = False




# === EVALUATION ===
def evaluate_aggressive(move, my_head, game_state, is_move_safe):

    delta = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    if not is_move_safe[move]: return -9999
    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}
    my_free, quality = calculate_free_space(new_head, game_state)
    enemy = [s for s in game_state['board']['snakes'] if s['id'] != game_state['you']['id']][0]
    enemy_free = calculate_enemy_free_space(enemy['body'][0], game_state)
    score = (my_free * 3) - (enemy_free * 4) + (quality * 1.5)
    cx, cy = game_state['board']['width']//2, game_state['board']['height']//2
    score += (10 - abs(new_head['x'] - cx) - abs(new_head['y'] - cy))
    my_length = game_state['you']['length']
    risk = is_true_head_on_risky(move, my_head, my_length, game_state)
    if risk == "death": score -= 1000
    elif risk == "neutral_risk": score -= 500
    elif risk == "advantage": score += 50
    return score

def evaluate_recovery(move, my_head, game_state, is_move_safe):
    delta = { "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0) }
    if not is_move_safe[move]: return -9999

    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}
    my_free_space, quality_score = calculate_free_space(new_head, game_state)

    food_list = game_state['board']['food']
    food_score = 0

    if food_list:
        closest_food = min(food_list, key=lambda f: abs(new_head['x'] - f['x']) + abs(new_head['y'] - f['y']))
        my_food_distance = abs(new_head['x'] - closest_food['x']) + abs(new_head['y'] - closest_food['y'])

        enemy_snakes = [s for s in game_state['board']['snakes'] if s['id'] != game_state['you']['id']]
        enemy_head = enemy_snakes[0]['body'][0]
        enemy_food_distance = abs(enemy_head['x'] - closest_food['x']) + abs(enemy_head['y'] - closest_food['y'])

        if my_food_distance + 1 <= enemy_food_distance:
            food_score = (20 - my_food_distance) * 10

    my_length = game_state["you"]["length"]
    head_on_result = is_true_head_on_risky(move, my_head, my_length, game_state)
    if head_on_result == "death":
        head_on_penalty = 1000
    elif head_on_result == "neutral_risk":
        head_on_penalty = 700
    else:
        head_on_penalty = 0

    score = food_score + my_free_space * 2 + quality_score - head_on_penalty
    return score

def evaluate_kill_mode(move, my_head, game_state, is_move_safe):
    delta = { "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0) }
    if not is_move_safe[move]: return -9999

    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}
    my_free_space, quality_score = calculate_free_space(new_head, game_state)

    # Enemy Raum berechnen
    enemy_snakes = [s for s in game_state['board']['snakes'] if s['id'] != game_state['you']['id']]
    enemy_head = enemy_snakes[0]['body'][0]
    enemy_free_space = calculate_enemy_free_space(enemy_head, game_state)

    # Head-on prüfen
    my_length = game_state["you"]["length"]
    head_on_result = is_true_head_on_risky(move, my_head, my_length, game_state)
    if head_on_result == "death":
        head_on_score = -1000
    elif head_on_result == "neutral_risk":
        head_on_score = -500
    elif head_on_result == "advantage":
        head_on_score = 200  # hier aktiver Bonus
    else:
        head_on_score = 0

    # Bewertung zusammenbauen
    score = (my_free_space * 2.5) - (enemy_free_space * 5) + (quality_score * 1.5) + head_on_score

    return score
# === MOVE ===
def move(game_state: typing.Dict) -> typing.Dict:
    board_width  = game_state['board']['width']
    board_height = game_state['board']['height']
    my_head = game_state['you']['body'][0]
    my_body = game_state['you']['body']
    my_id   = game_state['you']['id']
    snakes = game_state['board']['snakes']
    enemy  = next(s for s in snakes if s['id'] != my_id)

    # 1) Kollisionscheck (ohne den Schwanz als Hindernis)
    is_move_safe = {m: True for m in delta}
    avoid_collisions(my_head, my_body, snakes,
                     is_move_safe, board_width, board_height, my_id)
    safe_moves = [m for m, ok in is_move_safe.items() if ok]

    # 2) Schwanz-Ausnahme: Erlaube in die alte Schwanz-Zelle,
    #    außer dort liegt ein Apfel (dort würde der Schwanz stehen bleiben)
    tail = my_body[-1]
    food_set = {(f['x'], f['y']) for f in game_state['board']['food']}
    for m in safe_moves.copy():
        dx, dy = delta[m]
        nx, ny = my_head['x'] + dx, my_head['y'] + dy
        if (nx, ny) == (tail['x'], tail['y']):
            if (nx, ny) in food_set:
                safe_moves.remove(m)

    if not safe_moves:
        # Fallback: irgendeinen legalen Zug wählen
        for m, (dx, dy) in delta.items():
            nx, ny = my_head['x']+dx, my_head['y']+dy
            if 0 <= nx < board_width and 0 <= ny < board_height:
                return {"move": m}
        return {"move": "down"}

    # 3) Modus bestimmen
    mode = determine_mode(
        game_state['you']['length'],
        enemy['length'],
        game_state['you']['health']
    )

    # 4) Zug-Auswahl
    if mode in ("aggressive", "recovery", "kill_mode"):
        eval_fn = {
            'aggressive': evaluate_aggressive,
            'recovery':   evaluate_recovery,
            'kill_mode':  evaluate_kill_mode
        }[mode]
        best_move, best_val = None, -float('inf')
        alpha, beta = -float('inf'), float('inf')
        for m in safe_moves:
            val = evaluate_move_3ply(
                m, game_state, is_move_safe,
                eval_fn, alpha, beta, depth=3
            )
            if val > best_val:
                best_val, best_move = val, m
                alpha = max(alpha, val)
        chosen = best_move
    else:
        # Normal mode: freie Felder zählen, bei Gleichstand zufällig
        space_map = {
            m: calculate_free_space(
                   {'x': my_head['x'] + delta[m][0],
                    'y': my_head['y'] + delta[m][1]},
                   game_state
               )[0]
            for m in safe_moves
        }
        max_space = max(space_map.values())
        best = [m for m, sp in space_map.items() if sp == max_space]
        chosen = random.choice(best)

    return {"move": chosen}


# === START SERVER ===
if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})