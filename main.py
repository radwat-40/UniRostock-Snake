import typing
import copy
from collections import deque

# Bewegungsdeltas
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

# Transposition Table
transposition_table = {}

# === GAME START/END ===
def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER\n")

# === TRANSPO TABLE HELPERS ===
def lookup_in_tt(hash_key, depth, alpha, beta):
    entry = transposition_table.get(hash_key)
    if not entry or entry['depth'] < depth:
        return None

    val = entry['value']
    etype = entry['type']  # 'EXACT', 'LOWERBOUND', 'UPPERBOUND'

    if etype == 'EXACT':
        return val
    if etype == 'LOWERBOUND' and val > alpha:
        alpha = val
    if etype == 'UPPERBOUND' and val < beta:
        beta = val
    if alpha >= beta:
        return val
    return None


def store_in_tt(hash_key, depth, value, entry_type):
    transposition_table[hash_key] = {
        'depth': depth,
        'value': value,
        'type': entry_type  # 'EXACT', 'LOWERBOUND', 'UPPERBOUND'
    }

# === BOARD KEY ===
def board_to_key(state):
    parts = []
    for snake in state['board']['snakes']:
        parts.append(f"{snake['id']}:" + ",".join(f"{seg['x']}-{seg['y']}" for seg in snake['body']))
    food = ",".join(f"{f['x']}-{f['y']}" for f in state['board']['food'])
    parts.append("F:" + food)
    return "|".join(parts)

# === MODES ===
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

# === SIMULATION ===
def simulate_board_state(game_state, move_dict):
    new_state = copy.deepcopy(game_state)
    # Züge bewegen (Fallback 'down')
    for snake in new_state['board']['snakes']:
        sid = snake['id']
        move = move_dict.get(sid, 'down')
        dx, dy = delta[move]
        old_head = snake['body'][0]
        new_head = {"x": old_head["x"] + dx, "y": old_head["y"] + dy}
        snake['body'].insert(0, new_head)
    # Schwanz kürzen oder Food fressen
    food_positions = {(f['x'], f['y']) for f in new_state['board']['food']}
    for snake in new_state['board']['snakes']:
        head = snake['body'][0]
        if (head['x'], head['y']) in food_positions:
            new_state['board']['food'] = [f for f in new_state['board']['food']
                                          if not (f['x']==head['x'] and f['y']==head['y'])]
        else:
            snake['body'].pop()
    return new_state

# === FREIRAUM BERECHNUNG ===
def calculate_free_space(head, game_state, max_limit=50):
    w, h = game_state['board']['width'], game_state['board']['height']
    occupied = {(seg['x'], seg['y']) for s in game_state['board']['snakes'] for seg in s['body']}
    q = deque([(head['x'], head['y'])])
    vis = {(head['x'], head['y'])}
    free, quality = 0, 0
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]: pass
    while q and free < max_limit:
        x,y = q.popleft(); free+=1
        nbrs = 0
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and (nx,ny) not in occupied:
                nbrs+=1
                if (nx,ny) not in vis:
                    vis.add((nx,ny)); q.append((nx,ny))
        quality += nbrs
    return free, quality


def calculate_enemy_free_space(head, game_state, max_limit=50):
    w, h = game_state['board']['width'], game_state['board']['height']
    occupied = {(seg['x'], seg['y']) for s in game_state['board']['snakes'] for seg in s['body']}
    q = deque([(head['x'], head['y'])]); vis = {(head['x'], head['y'])}
    free = 0
    while q and free < max_limit:
        x,y = q.popleft(); free+=1
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and (nx,ny) not in occupied and (nx,ny) not in vis:
                vis.add((nx,ny)); q.append((nx,ny))
    return free

# === RISIKOPRÜFUNG ===
def is_true_head_on_risky(move, my_head, my_length, game_state):
    dx, dy = delta[move]
    nh = {"x": my_head["x"]+dx, "y": my_head["y"]+dy}
    w,h = game_state['board']['width'], game_state['board']['height']
    occupied = {(seg['x'], seg['y']) for s in game_state['board']['snakes'] for seg in s['body']}

    for snake in game_state['board']['snakes']:
        if snake['id']==game_state['you']['id']: continue
        eh = snake['body'][0]; el = snake['length']
        moves=[]
        for dx2,dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny=eh['x']+dx2, eh['y']+dy2
            if 0<=nx<w and 0<=ny<h and (nx,ny) not in occupied:
                moves.append((nx,ny))
        if (nh['x'],nh['y']) in moves:
            if el > my_length:
                return "death"
            elif el == my_length:
                return "neutral_risk"
            else:
                return "advantage"
    return "safe"

# === KOLLISIONSPRÜFUNG ===
def avoid_collisions(head, body, snakes, safe_map, w, h, my_id):
    if head['x']==0: safe_map['left']=False
    if head['x']==w-1: safe_map['right']=False
    if head['y']==0: safe_map['down']=False
    if head['y']==h-1: safe_map['up']=False
    for seg in body[1:]:
        if seg['x']==head['x']+1 and seg['y']==head['y']: safe_map['right']=False
        if seg['x']==head['x']-1 and seg['y']==head['y']: safe_map['left']=False
        if seg['x']==head['x'] and seg['y']==head['y']+1: safe_map['up']=False
        if seg['x']==head['x'] and seg['y']==head['y']-1: safe_map['down']=False
    for s in snakes:
        if s['id']==my_id: continue
        for seg in s['body']:
            if seg['x']==head['x']+1 and seg['y']==head['y']: safe_map['right']=False
            if seg['x']==head['x']-1 and seg['y']==head['y']: safe_map['left']=False
            if seg['x']==head['x'] and seg['y']==head['y']+1: safe_map['up']=False
            if seg['x']==head['x'] and seg['y']==head['y']-1: safe_map['down']=False

# === 3-PLY ALPHA-BETA ===
def evaluate_move_3ply(start_move: str,
                       game_state: typing.Dict,
                       is_move_safe: typing.Dict[str, bool],
                       evaluation_function: typing.Callable,
                       alpha: float,
                       beta: float,
                       depth: int) -> float:
    alpha_o, beta_o = alpha, beta
    key = board_to_key(game_state) + f"#d{depth}"
    cached = lookup_in_tt(key, depth, alpha, beta)
    if cached is not None:
        return cached

    my_id = game_state['you']['id']
    my_head = game_state['you']['body'][0]
    if not is_move_safe[start_move]:
        return -9999
    state1 = simulate_board_state(game_state, {my_id: start_move})
    enemies = [s for s in state1['board']['snakes'] if s['id']!=my_id]
    if not enemies:
        return evaluation_function(start_move, my_head, game_state, is_move_safe)
    enemy_id = enemies[0]['id']

    best = -float('inf')
    for e_move in delta:
        if best >= beta: break
        sim2 = simulate_board_state(game_state, {my_id: start_move, enemy_id: e_move})
        me = next(s for s in sim2['board']['snakes'] if s['id']==my_id)
        fh = me['body'][0]
        safe2 = {m: True for m in delta}
        avoid_collisions(fh, me['body'], sim2['board']['snakes'], safe2,
                         sim2['board']['width'], sim2['board']['height'], my_id)
        moves2 = [m for m,ok in safe2.items() if ok]
        if not moves2:
            resp = -9999
        else:
            resp = -float('inf')
            for m2 in moves2:
                if resp >= beta: break
                v = evaluation_function(m2, fh, sim2, safe2)
                if v > resp: resp = v
        if best == -float('inf') or resp < best:
            best = resp
        if best <= alpha:
            break
        alpha = max(alpha, best)

    if best <= alpha_o:
        et = 'UPPERBOUND'
    elif best >= beta_o:
        et = 'LOWERBOUND'
    else:
        et = 'EXACT'
    store_in_tt(key, depth, best, et)
    return best

# === EVALUATION FUNKS ===
def evaluate_aggressive(move, my_head, game_state, is_move_safe):
    if not is_move_safe[move]: return -9999
    dx,dy = delta[move]
    nh = {"x":my_head['x']+dx, 'y':my_head['y']+dy}
    my_f,qual = calculate_free_space(nh, game_state)
    en = [s for s in game_state['board']['snakes'] if s['id']!=game_state['you']['id']][0]
    ef = calculate_enemy_free_space(en['body'][0], game_state)
    score = my_f*3 - ef*4 + qual*1.5
    cx,cy = game_state['board']['width']//2, game_state['board']['height']//2
    score += (10 - abs(nh['x']-cx) - abs(nh['y']-cy))
    rl = is_true_head_on_risky(move, my_head, game_state['you']['length'], game_state)
    if rl == "death": score -= 1000
    elif rl == "neutral_risk": score -= 500
    elif rl == "advantage": score += 50
    return score


def evaluate_recovery(move, my_head, game_state, is_move_safe):
    if not is_move_safe[move]: return -9999
    dx,dy = delta[move]
    nh = {"x":my_head['x']+dx, 'y':my_head['y']+dy}
    fs,qs = calculate_free_space(nh, game_state)
    food = game_state['board']['food']
    fscr = 0
    if food:
        cf = min(food, key=lambda f: abs(nh['x']-f['x'])+abs(nh['y']-f['y']))
        dist_me = abs(nh['x']-cf['x'])+abs(nh['y']-cf['y'])
        en = [s for s in game_state['board']['snakes'] if s['id']!=game_state['you']['id']][0]
        dist_en = abs(en['body'][0]['x']-cf['x'])+abs(en['body'][0]['y']-cf['y'])
        if dist_me+1 <= dist_en:
            fscr = (20-dist_me)*10
    rl = is_true_head_on_risky(move, my_head, game_state['you']['length'], game_state)
    pen = 1000 if rl=="death" else 700 if rl=="neutral_risk" else 0
    return fscr + fs*2 + qs - pen


def evaluate_kill_mode(move, my_head, game_state, is_move_safe):
    if not is_move_safe[move]: return -9999
    dx,dy = delta[move]
    nh = {"x":my_head['x']+dx, 'y':my_head['y']+dy}
    my_f,qual = calculate_free_space(nh, game_state)
    en = [s for s in game_state['board']['snakes'] if s['id']!=game_state['you']['id']][0]
    ef = calculate_enemy_free_space(en['body'][0], game_state)
    rl = is_true_head_on_risky(move, my_head, game_state['you']['length'], game_state)
    hs = -1000 if rl=="death" else -500 if rl=="neutral_risk" else 200 if rl=="advantage" else 0
    return my_f*2.5 - ef*5 + qual*1.5 + hs

# === MOVE HANDLER ===
def move(game_state: typing.Dict) -> typing.Dict:
    w,h = game_state['board']['width'], game_state['board']['height']
    my = game_state['you']
    head, body = my['body'][0], my['body']
    lid, hp = my['length'], my['health']
    sid = my['id']
    snakes = game_state['board']['snakes']
    enemy = [s for s in snakes if s['id']!=sid][0]
    elen = enemy['length']

    occ = {(s['x'],s['y']) for s in snakes for s in s['body']}
    safe = {m: True for m in delta}
    avoid_collisions(head, body, snakes, safe, w, h, sid)
    moves = [m for m,ok in safe.items() if ok]
    if not moves:
        return {"move": "down"}

    mode = determine_mode(lid, elen, hp)
    if mode in ("aggressive","recovery","kill_mode"):
        fn = {'aggressive':evaluate_aggressive,
              'recovery':evaluate_recovery,
              'kill_mode':evaluate_kill_mode}[mode]
        best, bm = -float('inf'), None
        alpha, beta = -float('inf'), float('inf')
        for m in moves:
            v = evaluate_move_3ply(m, game_state, safe, fn, alpha, beta, depth=3)
            if v>best: best,bm=v,m; alpha=max(alpha,v)
        chosen = bm
    else:
        chosen = max(moves, key=lambda m: calculate_free_space(
            {'x': head['x']+delta[m][0], 'y': head['y']+delta[m][1]}, game_state
        )[0])

    print(f"Turn {game_state['turn']} Mode: {mode} Move: {chosen}")
    return {"move": chosen}

# === START SERVER ===
if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
