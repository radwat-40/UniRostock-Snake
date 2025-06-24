
import random
import typing
from collections import deque


#def spielgrundsätze---------------------------------------------------------------------------------------
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#736CCB",
        "head": "beluga",
        "tail": "curled"
    }



# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

#funktionen kollision--------------------------------------------------------------------------------------------------------

def is_head_on_risky(move, my_head, my_length, game_state):
    """
    Prüft, ob dieser Move zu einer Head-on-Kollision mit einem Gegner führen könnte.
    Gibt True zurück, wenn der Move riskant ist.
    """

    # Bewegungsrichtung definieren
    delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0)
    }
    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}

    # Alle Gegner durchgehen
    for snake in game_state['board']['snakes']:
        if snake["id"] == game_state["you"]["id"]:
            continue  # eigenen Snake überspringen

        enemy_head = snake["body"][0]
        enemy_length = snake["length"]

        # Prüfe: ist der gegnerische Kopf direkt neben dem neuen Head?
        distance = abs(new_head["x"] - enemy_head["x"]) + abs(new_head["y"] - enemy_head["y"])
        if distance == 1:
            # Gegner könnte in dasselbe Feld ziehen
            if enemy_length >= my_length:
                return True  # Riskant: Gegner ist gleich lang oder länger

    return False  # Kein Risiko gefunden

def evaluate_move(move, my_head, game_state, is_move_safe):
    """
    Bewertet den Move anhand von Raumgröße, Raumqualität und Health.
    Gibt einen numerischen Score zurück.
    """

    # Bewegungsrichtung definieren
    delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    # Sicherheitscheck: falls der Move nicht safe ist → sehr schlechter Score
    if not is_move_safe[move]:
        return -9999  # Unsicherer Move sofort extrem schlecht bewerten

    # Neue Position berechnen
    dx, dy = delta[move]
    new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}

    # Raumkontrolle berechnen (Flood-Fill)
    free_space, quality_score = calculate_free_space(new_head, game_state)

    # Gesundheitsfaktor: je weniger Health, desto wichtiger ist freier Raum
    health = game_state["you"]["health"]
    my_length = game_state["you"]["length"]

    if health < 40:
        health_factor = 1.5
    elif health < 20:
        health_factor = 2
    else:
        health_factor = 1

    # Gesamtscore berechnen
    score = (free_space * 2 + quality_score * 1.5) * health_factor
    if is_head_on_risky(move, my_head, my_length, game_state):
        score -= 100

    return score

def calculate_free_space(my_head, game_state, max_limit=50):
    """
    Henrik: 
    Flood-Fill Algorithmus mit Raumqualität.
    Gibt zwei Werte zurück:
    - free_space: Anzahl erreichbarer Felder
    - quality_score: Summe der freien Nachbarn (Raumqualität)
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    # Alle belegten Felder sammeln (eigener Körper + alle Gegner)
    occupied = set()
    for snake in game_state['board']['snakes']:
        for segment in snake['body']:
            occupied.add((segment['x'], segment['y']))

    # Setup BFS
    queue = deque()
    visited = set()
    queue.append((my_head['x'], my_head['y']))
    visited.add((my_head['x'], my_head['y']))

    free_space = 0
    quality_score = 0

    # Nachbarschaftsrichtungen
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        free_space += 1

        # Zähle freie Nachbarn für Qualitätsbewertung
        free_neighbors = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_width and 0 <= ny < board_height:
                if (nx, ny) not in occupied:
                    free_neighbors += 1
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        quality_score += free_neighbors

        # Früher Abbruch, um Performance zu sichern
        if free_space >= max_limit:
            break

    return free_space, quality_score

def avoid_self_collision(my_head, my_body, is_move_safe): #henrik -nicht in sich selber fahren 
    for segment in my_body[1:]:
        if segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]:
            is_move_safe["right"] = False
        if segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]:
            is_move_safe["left"] = False
        if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1:
            is_move_safe["up"] = False
        if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1:
            is_move_safe["down"] = False

def avoid_enemy_collision(my_head, snakes, is_move_safe, my_id):
    for snake in snakes:
        if snake["id"] == my_id:
            continue  #henrik: eigenen Körper überspringen, damit keine doppellungen
        for segment in snake["body"]:
            if segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]:
                is_move_safe["right"] = False
            if segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]:
                is_move_safe["left"] = False
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1:
                is_move_safe["up"] = False
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1:
                is_move_safe["down"] = False

def select_move_in_recovery_mode(my_head, food_list, is_move_safe, safe_moves): #bestimmt welcher move im recovbery mode am betsen ist 
    target_food = find_closest_food(my_head, food_list)

    if target_food:
        if target_food['x'] > my_head['x'] and is_move_safe["right"]:
            return "right"
        elif target_food['x'] < my_head['x'] and is_move_safe["left"]:
            return "left"
        elif target_food['y'] > my_head['y'] and is_move_safe["up"]:
            return "up"
        elif target_food['y'] < my_head['y'] and is_move_safe["down"]:
            return "down"
        else:
            return random.choice(safe_moves)
    else:
        return random.choice(safe_moves)

def determine_mode(my_length, enemy_length, my_health):  #Henrik: Legt fest, in welchem Modus wir uns gerade befinden:
    if my_health < 40:
        return "emergency" #Maximal defensiv bei hoher Gefahr.
    elif my_length >= enemy_length + 4:
        return "kill_mode"  # Neuer Modus: Head-on Angriff suchen
    elif my_length >= enemy_length + 2:
        return "aggressive" #Raum kontrollieren gegner in die enge treiben, wenn wir sicher länger sind.
    elif my_length <= enemy_length + 1:
        return "recovery" #Nahrung suchen, wenn Health niedrig oder Gegner uns einholt.
    else:
        return "normal" #Sicher spielen, Raum kontrollieren.

def find_closest_food(my_head, food_list):
    """
    Henrik: sobald recovery mode aktiviert ist muss essend gefunden werden
    Findet das nächste Food basierend auf der Manhattan-Distanz.
    Gibt die Position des am nächsten gelegenem Food zurück.
    """
    if not food_list:
        return None  # Kein Food vorhanden

    closest_food = None
    min_distance = float('inf') #wert wird auf unendlich gesetzt, damit sobald nächstes food hinzugefügt wird dieses "gejagt" wird 

    for food in food_list:
        distance = abs(my_head["x"] - food["x"]) + abs(my_head["y"] - food["y"])
        if distance < min_distance:
            min_distance = distance
            closest_food = food

    return closest_food

def choose_least_bad_move(my_head, my_neck, is_move_safe):
    #soll den am wenigsten schlechten move auswählen, becorzugt die Bewegungsrichtung
    # Bestimme aktuelle Bewegungsrichtung anhand der Position von Kopf und Hals
    if my_neck["x"] < my_head["x"]:
        forward = "right"
    elif my_neck["x"] > my_head["x"]:
        forward = "left"
    elif my_neck["y"] < my_head["y"]:
        forward = "up"
    else:
        forward = "down"

    # Liste aller sicheren Moves
    safe_moves = [move for move, safe in is_move_safe.items() if safe]

    # Bevorzuge den Move geradeaus (sofern noch sicher)
    if forward in safe_moves:
        return forward

    # Wenn Vorwärtsrichtung blockiert ist, wähle zufällig aus den übrigen sicheren Moves
    return random.choice(safe_moves)

def calculate_enemy_free_space(enemy_head, game_state, max_limit=50):
    """
    Flood-Fill für gegnerische Snake von einer bestimmten Position aus.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    occupied = set()
    for snake in game_state['board']['snakes']:
        for segment in snake['body']:
            occupied.add((segment['x'], segment['y']))

    queue = deque()
    visited = set()
    queue.append((enemy_head['x'], enemy_head['y']))
    visited.add((enemy_head['x'], enemy_head['y']))

    free_space = 0
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        free_space += 1

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_width and 0 <= ny < board_height:
                if (nx, ny) not in occupied and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        if free_space >= max_limit:
            break

    return free_space

def select_move_in_kill_mode(my_head, my_neck, game_state, is_move_safe):
    delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    safe_moves = [move for move, safe in is_move_safe.items() if safe]

    best_score = -99999
    best_move = None

    for move in safe_moves:
        dx, dy = delta[move]
        new_head = {"x": my_head["x"] + dx, "y": my_head["y"] + dy}

        # Head-on Check bleibt erhalten:
        for snake in game_state['board']['snakes']:
            if snake["id"] == game_state["you"]["id"]:
                continue
            enemy_head = snake["body"][0]
            distance = abs(new_head["x"] - enemy_head["x"]) + abs(new_head["y"] - enemy_head["y"])
            if distance == 1:
                return move  # Direkter Head-on sofort nutzen

        # Jetzt Enemy Flood-Fill simulieren
        enemy_snakes = [s for s in game_state['board']['snakes'] if s["id"] != game_state["you"]["id"]]
        enemy_free_space = calculate_enemy_free_space(enemy_snakes[0]["body"][0], game_state)

        # Bewertungsfunktion: Je weniger Raum der Gegner hat, desto besser für uns
        score = 100 - enemy_free_space  # je kleiner enemy_space, desto höher score

        if score > best_score:
            best_score = score
            best_move = move

    # Falls keine Head-on Chance, nimm den besten Druck-Move
    if best_move:
        return best_move

    # Fallback: immer noch stabil bleiben
    return choose_least_bad_move(my_head, my_neck, is_move_safe)

#def simulate_enemy_responses(my_head, game_state)
#Zweck: Simuliert für einen Zug die möglichen Antworten des Gegners.
#Nur leichter Lookahead (max. 1–2 Züge) → bleibt schnell.
#Hilft uns, riskante Situationen früh zu erkennen.

#BONUS: defis_enemy_trapped(enemy_head, game_state)
#Zweck: Erkennt, ob der Gegner nur noch wenig Raum zur Verfügung hat.
#Ermöglicht aktives Einsperren → führt zu automatischem Sieg.






#hier alles was jeden zug aufgerufen wird zum bewegen----------------------------------------------------------------------------
def move(game_state: typing.Dict) -> typing.Dict:
    board_width = game_state['board']['width'] #definieren der breite des Spielfeldes um später dareuf zurük zu greifen HR
    board_height = game_state['board']['height'] #definieren der höhe HR


    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False
    


    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds (done)
    #hier wird definiert, dass die schlange nicht in den Rend des spielfeldes "fährt"
    if my_head["x"] == 0: 
        is_move_safe["left"] = False  
    
    if my_head["x"] == board_width -1: 
        is_move_safe["right"] = False

    if my_head["y"] == 0: 
        is_move_safe["down"] = False
    
    if my_head["y"] == board_height -1: 
        is_move_safe["up"] = False
    

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body'] #henrik -my_body wird definiert für eigene körper kollision vermeiden 
    avoid_self_collision(my_head, my_body, is_move_safe) #henrik- nicht in sich selber fahren 

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    # opponents = game_state['board']['snakes']
    my_id = game_state['you']['id']
    snakes = game_state['board']['snakes']
    avoid_enemy_collision(my_head, snakes, is_move_safe, my_id)

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}


    # Henrik: Aktuellen Modus bestimmen:
    #vorher war hier random move jetzt update
    my_length = game_state['you']['length']
    my_health = game_state['you']['health']
    enemy_snakes = [s for s in snakes if s["id"] != my_id]
    enemy_length = enemy_snakes[0]["length"]
    current_mode = determine_mode(my_length, enemy_length, my_health)

    food_list = game_state['board']['food'] # food liste holen
    #Auswahl der nächsten Züge: 
    if current_mode == "recovery":
        next_move = select_move_in_recovery_mode(my_head, food_list, is_move_safe, safe_moves)
    elif current_mode == "kill_mode":
        next_move = select_move_in_kill_mode(my_head, game_state, is_move_safe, safe_moves)
    else: #wenn nicht niedrig hp dann das hier: moves 
        best_score = -99999 #schlechter score, damit alle neu hinzugefügten besser werden 
        best_move = None
        for move in safe_moves: #alle übrigen moves werden evaluiert 
            score = evaluate_move(move, my_head, game_state, is_move_safe)
            if score > best_score: #wenn neuer move besser ist als der bisher beste wird er zum neuen besten "Greedy Evaluation Pattern"
                best_score = score
                best_move = move
        next_move = best_move
#----------------------------------------------------------------------

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})





