import os
import random
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Constants
BOARD_SIZE = 11
MAX_HEALTH = 100

# Directions and vector mapping
MOVES = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0)
}

# Utility functions

def in_bounds(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def get_neighbors(pos):
    x, y = pos
    for dx, dy in MOVES.values():
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield (nx, ny)


def flood_fill(start, blocked):
    """Returns the number of reachable cells from start and count of branches."""
    visited = set([start])
    queue = deque([start])
    branches = 0
    while queue:
        cell = queue.popleft()
        neighbors = [n for n in get_neighbors(cell) if n not in blocked and n not in visited]
        if len(neighbors) > 1:
            branches += 1
        for n in neighbors:
            visited.add(n)
            queue.append(n)
    return len(visited), branches


def choose_move(data):
    board = data['board']
    you = data['you']
    foes = [s for s in board['snakes'] if s['id'] != you['id']]
    foe = foes[0]  # 1v1

    head = tuple(you['head'])
    health = you['health']
    length = len(you['body'])
    foe_head = tuple(foe['head'])
    foe_length = len(foe['body'])

    # Build base blocked set (including all snake bodies)
    blocked_base = set()
    for snake in board['snakes']:
        for part in snake['body']:
            blocked_base.add(tuple(part))

    # Tail handling: unless eating, the last tail segment becomes free
    tail = tuple(you['body'][-1])
    apple_positions = {tuple(a) for a in board['food']}

    # Determine mode
    if length < foe_length or health < 40:
        mode = 'food_hunter'
    elif length >= foe_length + 2:
        mode = 'aggressive'
    else:
        mode = 'neutral'

    best_score = float('-inf')
    best_moves = []

    # Evaluate each potential move
    for move, (dx, dy) in MOVES.items():
        new_head = (head[0] + dx, head[1] + dy)
        # Check wall collision
        if not in_bounds(*new_head):
            continue
        # Build dynamic blocked for this move
        blocked = set(blocked_base)
        # If not eating apple, tail moves -> free
        if new_head not in apple_positions:
            blocked.discard(tail)
        # Now if new head collides with body
        if new_head in blocked:
            continue

        # Flood-fill for both snakes
        my_space, my_branches = flood_fill(new_head, blocked)
        foe_space, _ = flood_fill(foe_head, blocked)

        # Distance metrics
        apple_dist = (min(abs(new_head[0]-ax) + abs(new_head[1]-ay) for ax, ay in apple_positions)
                      if apple_positions else BOARD_SIZE*2)
        foe_dist = abs(new_head[0] - foe_head[0]) + abs(new_head[1] - foe_head[1])

        # Scoring
        score = 0
        if mode == 'food_hunter':
            score += (BOARD_SIZE*2 - apple_dist) * 1.5
            score += my_space * 1.0
            score -= foe_dist * 1.0
        elif mode == 'aggressive':
            score += my_space * 2
            score -= foe_space * 1.5
            # Head-on bonus
            if foe_dist == 1 and length > foe_length:
                score += 50
        else:  # neutral
            score += my_space * 1.2
            score += my_branches * 1.0
            score += max(0, (BOARD_SIZE*2 - apple_dist)) * 0.5

        # Track best
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    # If no safe move, fallback
    if not best_moves:
        return random.choice(list(MOVES.keys()))
    return random.choice(best_moves)

# API endpoints
@app.route('/', methods=['GET'])
def handle_index():
    return jsonify({
        'apiversion': '1',
        'author': 'Püppchen',
        'color': '#ff0000',
        'head': 'evil',
        'tail': 'curled'
    })

@app.route('/start', methods=['POST'])
def handle_start():
    return jsonify({})

@app.route('/move', methods=['POST'])
def handle_move():
    move = choose_move(request.get_json())
    return jsonify({'move': move})

@app.route('/end', methods=['POST'])
def handle_end():
    return jsonify({})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port)
