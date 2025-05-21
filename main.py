# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "YNS",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

def count_free_space(board, start, body):
    width = board['width']
    height = board['height']
    visited = set()
    queue = [ (start['x'], start['y']) ]
    body_set = set((segment['x'], segment['y']) for segment in body)
    count = 0

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or (x, y) in body_set:
            continue
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        visited.add((x, y))
        count += 1
        # Nachbarfelder hinzufügen
        for dx, dy in [ (0,1), (0,-1), (1,0), (-1,0) ]:
            queue.append( (x+dx, y+dy) )
    return count

def simulate_next_safe_moves(board, body, move, next_positions, depth=2):
    """
    Simulate the snake's move for a given depth and return the minimum number of safe moves found in the lookahead.
    """
    head = next_positions[move]
    new_body = [head] + body[:-1]
    safe_count = 0

    # Generate possible next moves
    possible_moves = {
        "up":    {"x": head["x"],     "y": head["y"] + 1},
        "down":  {"x": head["x"],     "y": head["y"] - 1},
        "left":  {"x": head["x"] - 1, "y": head["y"]},
        "right": {"x": head["x"] + 1, "y": head["y"]},
    }
    body_set = set((segment["x"], segment["y"]) for segment in new_body[1:])
    board_width = board['width']
    board_height = board['height']

    safe_moves = []
    for m, pos in possible_moves.items():
        if (
            0 <= pos["x"] < board_width and
            0 <= pos["y"] < board_height and
            (pos["x"], pos["y"]) not in body_set
        ):
            safe_moves.append(m)

    safe_count = len(safe_moves)

    # Recursive lookahead
    if depth > 1 and safe_moves:
        min_next = float('inf')
        for m in safe_moves:
            next_safe = simulate_next_safe_moves(board, new_body, m, possible_moves, depth-1)
            if next_safe < min_next:
                min_next = next_safe
        safe_count = min(safe_count, min_next)

    return safe_count

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"
    my_body = game_state['you']['body']
    tail = my_body[-1]
    health = game_state["you"]["health"]

    next_positions = {
        "up":    {"x": my_head["x"],     "y": my_head["y"] + 1},
        "down":  {"x": my_head["x"],     "y": my_head["y"] - 1},
        "left":  {"x": my_head["x"] - 1, "y": my_head["y"]},
        "right": {"x": my_head["x"] + 1, "y": my_head["y"]},
    }

    # Prevent moving backwards
    if my_neck["x"] < my_head["x"]:
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:
        is_move_safe["up"] = False

    # Prevent your Battlesnake from moving out of bounds (done)
    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # Prevent your Battlesnake from colliding with itself and avoid small spaces
    body_set = set((segment["x"], segment["y"]) for segment in my_body[1:])
    min_required_space = len(my_body)
    filtered_moves = []
    for move, pos in next_positions.items():
        if (pos["x"], pos["y"]) in body_set:
            is_move_safe[move] = False
            continue
        space = count_free_space(game_state['board'], pos, my_body)
        if space >= min_required_space:
            filtered_moves.append(move)
        else:
            is_move_safe[move] = False

    safe_moves = filtered_moves

    # Avoid corners and edges if possible
    def is_corner(pos):
        return (pos["x"] == 0 or pos["x"] == board_width - 1) and (pos["y"] == 0 or pos["y"] == board_height - 1)
    def is_edge(pos):
        return pos["x"] == 0 or pos["x"] == board_width - 1 or pos["y"] == 0 or pos["y"] == board_height - 1

    non_corner_moves = [move for move in safe_moves if not is_corner(next_positions[move])]
    non_edge_moves = [move for move in safe_moves if not is_edge(next_positions[move])]

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    food = game_state['board']['food']
    need_food = (health < 40 or len(food) == 0)  # You can adjust the health threshold

    def manhattan(a, b):
        return abs(a['x'] - b['x']) + abs(a['y'] - b['y'])

    # Prioritize moves: non-corner > non-edge > any safe move
    if non_corner_moves:
        candidate_moves = non_corner_moves
    elif non_edge_moves:
        candidate_moves = non_edge_moves
    else:
        candidate_moves = safe_moves

    # --- Efficient lookahead and food logic ---
    lookahead_depth = 1  # Keep this low for performance
    lookahead_moves = []
    for move in candidate_moves:
        safe_future = simulate_next_safe_moves(game_state['board'], my_body, move, next_positions, depth=lookahead_depth)
        if safe_future > 0:
            lookahead_moves.append(move)

    candidate_moves = lookahead_moves if lookahead_moves else candidate_moves

    # Prefer moves that get closer to food if food exists and you need food, otherwise follow tail or pick random
    if food and need_food:
        closest_food = min(food, key=lambda f: manhattan(my_head, f))
        min_dist = float('inf')
        food_move = None
        for move in candidate_moves:
            new_head = next_positions[move]
            dist = manhattan(new_head, closest_food)
            if dist < min_dist:
                min_dist = dist
                food_move = move
        if food_move:
            next_move = food_move
        else:
            next_move = random.choice(candidate_moves)
    else:
        # If not in urgent need of food, follow tail if possible, else random
        min_dist_tail = float('inf')
        tail_move = None
        for move in candidate_moves:
            new_head = next_positions[move]
            dist = manhattan(new_head, tail)
            if dist < min_dist_tail:
                min_dist_tail = dist
                tail_move = move
        if tail_move:
            next_move = tail_move
        else:
            next_move = random.choice(candidate_moves)

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
