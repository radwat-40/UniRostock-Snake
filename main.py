import typing
import random
from collections import deque

delta = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0)
}

# berechnet naechsten zug
#Funktionen:
def detect_dead_end(start, board, snakes, depth_limit=10):
    """
    Führt eine Flood-Fill-Analyse durch, um zu erkennen, ob ein Pfad in eine Sackgasse führt.
    Gibt True zurück, wenn es wahrscheinlich eine Sackgasse ist, False sonst.
    """
    from collections import deque

    visited = set()
    queue = deque()
    queue.append((start['x'], start['y'], 0))
    visited.add((start['x'], start['y']))
    
    board_width, board_height = board['width'], board['height']
    occupied = {(seg['x'], seg['y']) for s in snakes for seg in s['body']}
    
    reachable_tiles = 0
    for_count = 0  # Sicherheitsmechanismus, um Abbrüche zu verhindern
    
    while queue and for_count < 100:
        x, y, depth = queue.popleft()
        for_count += 1
        if depth >= depth_limit:
            continue
        reachable_tiles += 1
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board_width and 0 <= ny < board_height and 
                (nx, ny) not in visited and (nx, ny) not in occupied):
                visited.add((nx, ny))
                queue.append((nx, ny, depth + 1))
    
    # Sackgasse, wenn zu wenig freie Felder erreichbar sind
    return reachable_tiles < depth_limit // 2



def move(game_state: typing.Dict) -> typing.Dict:
    board = game_state['board']
    you = game_state['you']

    my_head = you['body'][0]
    my_health = you['health']
    my_length = you['length']
    board_width = board['width']
    board_height = board['height']
    enemy = [s for s in board['snakes'] if s['id'] != you['id']][0]

    # Moduslogik nach Länge und Leben
    mode = "neutral"
    if my_health < 40 or my_length <= enemy['length']:
        mode = "food_hunter"
    elif my_length >= enemy['length'] + 2:
        mode = "aggressive"

    safe_moves = []
    for m, (dx, dy) in delta.items():
        new_x = my_head['x'] + dx
        new_y = my_head['y'] + dy

        # Spielfeldgrenzen prüfen
        if not (0 <= new_x < board_width and 0 <= new_y < board_height):
            continue

        # Belegte Felder vermeiden
        ate_food = any(f['x'] == new_x and f['y'] == new_y for f in board['food'])
        ignore_tail = you if not ate_food else None
        if is_occupied(new_x, new_y, board['snakes'], ignore_tail=ignore_tail):
            continue

        # Head-on-Risiko prüfen
        is_risky_head_on = False
        for other in board['snakes']:
            if other['id'] == you['id']:
                continue
            enemy_head = other['body'][0]
            if abs(enemy_head['x'] - new_x) + abs(enemy_head['y'] - new_y) == 1:
                if my_length <= len(other['body']):
                    is_risky_head_on = True
                    break

        if is_risky_head_on:
            continue

        # Dead-End-Vermeidung
        new_head = {'x': new_x, 'y': new_y}
        if detect_dead_end(new_head, board, board['snakes']):
            continue

        safe_moves.append(m)

    if not safe_moves:
        return {"move": "up"}  # Notfallzug

    best_score = -9999
    best_move = safe_moves[0]

    for move in safe_moves:
        dx, dy = delta[move]
        new_head = {'x': my_head['x'] + dx, 'y': my_head['y'] + dy}
        flood_score, quality = flood_fill(new_head, board, limit=50)
        score = 0

        if mode == "food_hunter":
            dist = closest_food_distance(new_head, board['food'])
            if dist is not None:
                score += (50 - dist) * 2
            score += flood_score + quality

        elif mode == "aggressive":
            enemy_score, _ = flood_fill(enemy['body'][0], board, limit=50)
            score += flood_score * 2 - enemy_score * 3 + quality

        else:
            score += flood_score + quality

        if score > best_score:
            best_score = score
            best_move = move

    return {"move": best_move}


# prueft feld belegung
def is_occupied(x, y, snakes, ignore_tail=None):
    for s in snakes:
        for i, b in enumerate(s['body']):
            if s == ignore_tail and i == len(s['body']) - 1:
                continue  # Tail ignorieren, wenn erlaubt
            if b['x'] == x and b['y'] == y:
                return True
    return False

# berechnet freien raum
def flood_fill(start: dict, board: dict, limit: int = 50):
    visited = set()
    q = deque()
    q.append((start['x'], start['y']))
    visited.add((start['x'], start['y']))

    board_w, board_h = board['width'], board['height']
    snakes = board['snakes']
    count = 0
    quality = 0

    while q and count < limit:
        x, y = q.popleft()
        count += 1
        free_neighbors = 0

        for dx, dy in delta.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_w and 0 <= ny < board_h:
                if (nx, ny) not in visited and not is_occupied(nx, ny, snakes):
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    free_neighbors += 1

        quality += free_neighbors

    return count, quality

# misst futter abstand
def closest_food_distance(pos, food_list):
    if not food_list:
        return None
    return min(abs(pos['x'] - f['x']) + abs(pos['y'] - f['y']) for f in food_list)

# gibt snake infos
def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "flood-fighter",
        "color": "#69420",
        "head": "trans-rights-scarf",
        "tail": "bolt"
    }

# spielstartmeldung
def start(game_state: typing.Dict):
    print("Game started")

# spielendmeldung
def end(game_state: typing.Dict):
    print("Game over")

# serverstart logik
if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
