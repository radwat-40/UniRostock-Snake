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
ZOB_SNAKE = {}  # wird in start() pro Snake-ID initialisiert
ZOB_FOOD  = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
current_hash = 0  # globaler 64-Bit Hash

# === INFO ===
def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "team-WIN",
        "color": "#97FF3C",
        "head": "gamer",
        "tail": "bolt"
    }

transposition_table = {}

def init_hash(state):
    h = 0
    for snake in state['board']['snakes']:
        sid = snake['id']
        if sid not in ZOB_SNAKE:
            ZOB_SNAKE[sid] = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
        for seg in snake['body']:
            h ^= ZOB_SNAKE[sid][seg['x']][seg['y']]
    for f in state['board']['food']:
        h ^= ZOB_FOOD[f['x']][f['y']]
    return h

def start(game_state: typing.Dict):
    global current_hash
    for snake in game_state['board']['snakes']:
        sid = snake['id']
        if sid not in ZOB_SNAKE:
            ZOB_SNAKE[sid] = [[random.getrandbits(64) for _ in range(11)] for _ in range(11)]
    current_hash = init_hash(game_state)
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")

def apply_moves(game_state: typing.Dict, move_dict: typing.Dict[str,str]) -> typing.List[tuple]:
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
        current_hash ^= ZOB_SNAKE[sid][head['x']][head['y']]
        current_hash ^= ZOB_SNAKE[sid][new_head['x']][new_head['y']]
        snake['body'].insert(0, new_head)

        if (new_head["x"], new_head["y"]) in food_set:
            current_hash ^= ZOB_FOOD[new_head['x']][new_head['y']]
            for i, f in enumerate(board['food']):
                if (f['x'], f['y']) == (new_head['x'], new_head['y']):
                    removed = board['food'].pop(i)
                    changes.append((sid, True, removed))
                    break
        else:
            tail = snake['body'].pop()
            current_hash ^= ZOB_SNAKE[sid][tail['x']][tail['y']]
            changes.append((sid, False, tail))

    return changes

def undo_moves(game_state: typing.Dict, changes: typing.List[tuple]):
    global current_hash
    board = game_state['board']
    for sid, ate, seg in reversed(changes):
        snake = next(s for s in board['snakes'] if s['id'] == sid)
        head = snake['body'][0]
        snake['body'].pop(0)
        current_hash ^= ZOB_SNAKE[sid][head['x']][head['y']]
        if ate:
            board['food'].append(seg)
            current_hash ^= ZOB_FOOD[seg['x']][seg['y']]
        else:
            snake['body'].append(seg)
            current_hash ^= ZOB_SNAKE[sid][seg['x']][seg['y']]

def evaluate_move_2ply(start_move: str,
                       game_state: typing.Dict,
                       is_move_safe: typing.Dict[str, bool],
                       evaluation_function: typing.Callable,
                       alpha: float,
                       beta: float,
                       depth: int) -> float:
    alpha_orig, beta_orig = alpha, beta
    key = (current_hash, depth)
    cached = lookup_in_tt(key, depth, alpha, beta)
    if cached is not None:
        return cached

    best_score = -float("inf")
    move_dict = {}
    for snake in game_state["board"]["snakes"]:
        sid = snake["id"]
        if sid == game_state["you"]["id"]:
            move_dict[sid] = start_move
        else:
            enemy_snake = snake
            ex, ey = enemy_snake['body'][0]['x'], enemy_snake['body'][0]['y']
            enemy_options = []
            for dir_key, (dx, dy) in delta.items():
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < game_state['board']['width'] and 0 <= ny < game_state['board']['height']:
                    if (nx, ny) not in {(seg['x'], seg['y']) for s in game_state['board']['snakes'] for seg in s['body']}:
                        enemy_options.append(dir_key)
            move_dict[sid] = random.choice(enemy_options) if enemy_options else "up"

    changes = apply_moves(game_state, move_dict)

    if depth == 0:
        best_score = evaluation_function(start_move, game_state["you"]["body"][0], game_state, is_move_safe)
    else:
        for next_move in delta:
            score = evaluate_move_2ply(next_move, game_state, is_move_safe, evaluation_function, alpha, beta, depth - 1)
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

    undo_moves(game_state, changes)

    etype = 'EXACT'
    if best_score <= alpha_orig:
        etype = 'UPPERBOUND'
    elif best_score >= beta_orig:
        etype = 'LOWERBOUND'
    store_in_tt(key, depth, best_score, etype)
    return best_score
