

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

# === GAME START ===
def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")
#----------------Funktionen------------------------------
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

def board_to_key(state):
    parts = []
    for snake in state['board']['snakes']:
        parts.append(f"{snake['id']}:" + ",".join(f"{seg['x']}-{seg['y']}" for seg in snake['body']))
    food = ",".join(f"{f['x']}-{f['y']}" for f in state['board']['food'])
    parts.append("F:" + food)
    return "|".join(parts)

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
        return "recovery"

def simulate_board_state(game_state, move_dict):
    """
    Erzeugt einen neuen Spielzustand basierend auf den übergebenen Zügen.
    Berücksichtigt: Bewegung, Apfelfressen, Tail-Entfernung, Snake-Death.
    """
    new_state = copy.deepcopy(game_state)
    board_width = new_state['board']['width']
    board_height = new_state['board']['height']
    food_positions = {(f['x'], f['y']) for f in new_state['board']['food']}

    # Schritt 1: Alle Snakes bewegen
    new_heads = {}
    for snake in new_state['board']['snakes']:
        sid = snake['id']
        move = move_dict.get(sid, "up")
        dx, dy = delta[move]
        old_head = snake['body'][0]
        new_head = {'x': old_head['x'] + dx, 'y': old_head['y'] + dy}
        snake['body'].insert(0, new_head)
        snake['health'] -= 1  # health sinkt um 1 pro Zug
        new_heads[sid] = new_head

    # Schritt 2: Essen prüfen
    new_food = []
    for f in new_state['board']['food']:
        eaten = False
        for snake in new_state['board']['snakes']:
            if snake['body'][0]['x'] == f['x'] and snake['body'][0]['y'] == f['y']:
                snake['health'] = 100
                eaten = True
                break
        if not eaten:
            new_food.append(f)
    new_state['board']['food'] = new_food

    # Schritt 3: Tail entfernen, wenn kein Apfel gegessen
    for snake in new_state['board']['snakes']:
        head = snake['body'][0]
        if (head['x'], head['y']) not in food_positions:
            snake['body'].pop()

    # Schritt 4: Kollisionen prüfen
    occupied = {}  # Positionen nach Bewegung
    for snake in new_state['board']['snakes']:
        for i, segment in enumerate(snake['body']):
            pos = (segment['x'], segment['y'])
            if pos not in occupied:
                occupied[pos] = []
            occupied[pos].append((snake['id'], i))

    surviving_snakes = []
    for snake in new_state['board']['snakes']:
        head = snake['body'][0]
        sid = snake['id']
        # Wandkollision
        if not (0 <= head['x'] < board_width and 0 <= head['y'] < board_height):
            continue
        # Health-Kollaps
        if snake['health'] <= 0:
            continue
        # Kollision mit Körper
        count = occupied.get((head['x'], head['y']), [])
        if len(count) > 1:
            # Head-to-head: nur Snake mit längstem Körper überlebt
            max_len = max(len([s for s in new_state['board']['snakes'] if s['id'] == sid][0]['body']) for sid2, _ in count)
            if len(snake['body']) < max_len:
                continue
            elif len(snake['body']) == max_len:
                # beide sterben, wird oben durch Länge >1 behandelt
                pass
        surviving_snakes.append(snake)

    new_state['board']['snakes'] = surviving_snakes
    return new_state


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
        enemy_moves = []
        for ex, ey in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = enemy_head["x"] + ex, enemy_head["y"] + ey
            if 0 <= nx < board_width and 0 <= ny < board_height and (nx, ny) not in occupied:
                enemy_moves.append((nx, ny))
        if (new_head["x"], new_head["y"]) in enemy_moves:
            if enemy_length > my_length:
                return "death"
            elif enemy_length == my_length:
                return "neutral_risk"
            else:
                return "advantage"
    return "safe"

def avoid_collisions(my_head, my_body, snakes, is_move_safe, board_width, board_height, my_id):
    # Wand-Kollision
    if my_head["x"] == 0: is_move_safe["left"] = False
    if my_head["x"] == board_width - 1: is_move_safe["right"] = False
    if my_head["y"] == 0: is_move_safe["down"] = False
    if my_head["y"] == board_height - 1: is_move_safe["up"] = False

    # Eigener Körper
    for segment in my_body[1:]:
        if segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]: is_move_safe["right"] = False
        if segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]: is_move_safe["left"] = False
        if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1: is_move_safe["up"] = False
        if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1: is_move_safe["down"] = False

    # Gegnerkörper
    for snake in snakes:
        if snake["id"] == my_id: continue
        for segment in snake["body"]:
            if segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]: is_move_safe["right"] = False
            if segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]: is_move_safe["left"] = False
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1: is_move_safe["up"] = False
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1: is_move_safe["down"] = False

def evaluate_move_2ply(start_move: str,
                       game_state: typing.Dict,
                       is_move_safe: typing.Dict[str, bool],
                       evaluation_function: typing.Callable,
                       alpha: float,
                       beta: float,
                       depth: int) -> float:
    alpha_orig, beta_orig = alpha, beta

    key = board_to_key(game_state) + f"#d{depth}"
    entry = lookup_in_tt(key, depth, alpha, beta)
    if entry is not None:
        return entry

    my_id = game_state['you']['id']
    my_head = game_state['you']['body'][0]

    if not is_move_safe[start_move]:
        return -9999

    enemies = [s for s in game_state['board']['snakes'] if s['id'] != my_id]
    if not enemies:
        # keine Gegner? einfache Bewertung nach deinem Zug
        move_dict = {my_id: start_move}
        state_after = simulate_board_state(game_state, move_dict)
        return evaluation_function(start_move, my_head, state_after, is_move_safe)

    enemy_id = enemies[0]['id']
    worst_case_score = float("inf")

    for enemy_move in delta:
        move_dict = {
            my_id: start_move,
            enemy_id: enemy_move
        }
        state_after_both = simulate_board_state(game_state, move_dict)
        score = evaluation_function(start_move, my_head, state_after_both, is_move_safe)

        if score < worst_case_score:
            worst_case_score = score

        # Alpha-Beta-Pruning (Minimizing opponent response)
        if worst_case_score <= alpha:
            break
        beta = min(beta, worst_case_score)

    # Speichern in Transposition Table
    if worst_case_score <= alpha_orig:
        etype = 'UPPERBOUND'
    elif worst_case_score >= beta_orig:
        etype = 'LOWERBOUND'
    else:
        etype = 'EXACT'
    store_in_tt(key, depth, worst_case_score, etype)

    return worst_case_score





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
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    my_head = game_state['you']['body'][0]
    my_body = game_state['you']['body']
    my_length = game_state['you']['length']
    my_health = game_state['you']['health']
    my_id = game_state['you']['id']
    snakes = game_state['board']['snakes']
    enemy = [s for s in snakes if s['id'] != my_id][0]
    enemy_length = enemy['length']

    # Occupied-Set einmalig
    occupied = {(seg['x'], seg['y']) for s in snakes for seg in s['body']}

    # Kollisionscheck
    is_move_safe = {m: True for m in delta}
    avoid_collisions(my_head, my_body, snakes, is_move_safe, board_width, board_height, my_id)
    safe_moves = [m for m, ok in is_move_safe.items() if ok]
    if not safe_moves:
        return {"move": "down"}

    mode = determine_mode(my_length, enemy_length, my_health)

    # Auswahl mit Alpha-Beta in 3-Ply
    if mode in ("aggressive", "recovery", "kill_mode"):
        eval_fn = {'aggressive': evaluate_aggressive,
                   'recovery': evaluate_recovery,
                   'kill_mode': evaluate_kill_mode}[mode]
        best_move = None
        best_val = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        for m in safe_moves:
            # Depth=3, weil wir 3 Ply Lookahead machen
            val = evaluate_move_2ply(m,
                         game_state,
                         is_move_safe,
                         eval_fn,
                         alpha,
                         beta,
                         depth=2)

            if val > best_val:
                best_val, best_move = val, m
                alpha = max(alpha, val)
        chosen = best_move
    else:
        # Normal mode: freier Raum
        chosen = max(
            safe_moves,
            key=lambda m: calculate_free_space(
                {'x': my_head['x'] + delta[m][0], 'y': my_head['y'] + delta[m][1]},
                game_state
            )[0]
        )

    if not chosen:
        print("WARNING: Kein Move wurde gewählt – fallback auf ersten sicheren Zug.")
        chosen = safe_moves[0]

    print(f"Turn {game_state['turn']} Mode: {mode} Move: {chosen}")
    return {"move": chosen}

# === START SERVER ===
if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})