
import random
import typing


#def spielgrundsätze---------------------------------------------------------------------------------------
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
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

#funktionen kollision--------------------------------------------------------------------------------------------------------

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
            continue  #henrik eigenen Körper überspringen, damit keine doppellungen
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

#def calculate_free_space(my_head, game_state)
#Zweck: Bestimmt, wie viel freier Raum uns nach einem Zug noch zur Verfügung steht.
#Nutzt Flood-Fill (z. B. BFS), um Sackgassen zu vermeiden.
#Wichtig, um langfristig den Gegner in enge Bereiche zu drängen.

#def is_head_on_risky(my_head, my_length, enemy_heads, enemy_lengths)
#Zweck: Ermittelt, ob ein Kopf-an-Kopf-Kampf im nächsten Zug gefährlich wäre.
#Wichtig um tödliche Duelle nur einzugehen, wenn wir länger sind.

#def evaluate_move(move, game_state)
# Zweck: Bewertet jeden möglichen Zug mit einem Punktesystem.
#Bewertet u.a.:
#Freien Raum (Ergebnis von calculate_free_space())
#Gesundheitszustand (Health)
#Head-on-Sicherheit
#Entfernung zu Nahrung (nur wenn Nahrung nötig)
#Positionskontrolle




def determine_mode(my_length, enemy_length, my_health):  #Henrik: Legt fest, in welchem Modus wir uns gerade befinden:
    if my_health < 40:
        return "emergency" #Maximal defensiv bei hoher Gefahr.
    elif my_length <= enemy_length + 1:
        return "recovery" #Nahrung suchen, wenn Health niedrig oder Gegner uns einholt.
    elif my_length >= enemy_length + 3:
        return "aggressive" #Head-on suchen, wenn wir sicher länger sind.
    else:
        return "normal" #Sicher spielen, Raum kontrollieren.






#def ind_closest_food(my_head, food_list)
#Zweck: Findet gezielt die nächstliegende Nahrung, wenn wir sie brauchen.
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





#def simulate_enemy_responses(my_head, game_state)
#Zweck: Simuliert für einen Zug die möglichen Antworten des Gegners.
#Nur leichter Lookahead (max. 1–2 Züge) → bleibt schnell.
#Hilft uns, riskante Situationen früh zu erkennen.

#def choose_least_bad_move(is_move_safe)
#Zweck: Falls alle anderen Bewertungen unsicher sind:
#Wählt den sichersten noch verfügbaren Move.
#Verhindert, dass wir kampflos sterben.

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
    else:
        next_move = random.choice(safe_moves)
#----------------------------------------------------------------------

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})





