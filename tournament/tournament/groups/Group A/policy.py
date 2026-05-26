import numpy as np
from connect4.policy import Policy



TRAINING_GAMES = 1_000 # partidas de entrenamiento mount 
EPSILON = 0.2  # probabilidad de movimiento aleatorio en entrenamiento
LEARNING_RATE  = 0.1      # lr: qué tan grande es cada actualización de V(s)
DISCOUNT       = 0.95     # gamma: cuánto peso tienen los estados futuros
 
ROWS = 6
COLS = 7
EMPTY = 0

 
class ADPAgent(Policy): #agente oficial
 
    trained = False
    V_global = {}

    def mount(self, *args, **kwargs):

        # si ya entrenó, reutiliza
        if ADPAgent.trained:
            self.V = ADPAgent.V_global
            return

        self.V = {} #empieza con un diccionario vacio, el agente no sabe nada 
 
        for _ in range(TRAINING_GAMES): #el agente juega con el mismo  aprtidas aleatorias para aprender  
            board = np.zeros((ROWS, COLS), dtype=int) #crea un tablero vacio 
            path = [] #guarda todos los estados del tablero durante la partida 
            player = 1 #empieza el jugador 1 o max


            #loop del juego 
            while True: #aqui se juega la partida completa hasta que termine 
                cols_libres = _cols_libres(board) #busca columnas disponibles 
                if not cols_libres: #si no hay movimientos, empate 
                    break

                #exploracion =epsilon 
 
                if np.random.rand() < EPSILON: #a veces juega random  para aprender cosas nuevas 
                    #caso 1
                    accion = int(np.random.choice(cols_libres))
                else:
                    #caso 2 usa lo que ha aprendido v
                    accion = _mejor_accion(board, player, self.V, cols_libres)

                #guarda experiencia 
                #el estado del tablero  y quien jugo ( para aprneder despues)
                path.append((board.copy(), player))
                #simula la jugada 
                board = _colocar(board, accion, player)
                #verifica si termino 
                ganador = _get_winner(board)
                #si alguien ganó o empate termina la partida 
                if ganador != 0 or not _cols_libres(board):
                    break
                #cambia de jugador 
                player = -player
 

            ganador = _get_winner(board)
            #el agente aprende de toda la partid, to update recoore la parrida hacia atras 
            _td_update(path, ganador, self.V, LEARNING_RATE, DISCOUNT) #aca el agente aprende
 
        ADPAgent.V_global = self.V
        ADPAgent.trained = True

    def act(self, s: np.ndarray) -> int:

        # seguridad por si mount no fue llamado
        if not hasattr(self, "V"):
            self.V = {}

        cols_libres = _cols_libres(s)

        if not cols_libres:
            return 0

        return int(
            _mejor_accion(
                s,
                1,
                self.V,
                cols_libres
            )
        )


def _state_key(board):

    return str(board.reshape(ROWS * COLS))


def _cols_libres(board):

    return [
        c for c in range(COLS)
        if board[0, c] == EMPTY
    ]


def _colocar(board, col, player):

    nuevo = board.copy()

    for r in reversed(range(ROWS)):

        if nuevo[r, col] == EMPTY:

            nuevo[r, col] = player
            break

    return nuevo


def _get_winner(board):

    for r in range(ROWS):

        for c in range(COLS):

            p = board[r, c]

            if p == 0:
                continue

            # horizontal
            if (
                c + 3 < COLS
                and all(board[r, c+i] == p for i in range(4))
            ):
                return p

            # vertical
            if (
                r + 3 < ROWS
                and all(board[r+i, c] == p for i in range(4))
            ):
                return p

            # diagonal derecha
            if (
                r + 3 < ROWS
                and c + 3 < COLS
                and all(board[r+i, c+i] == p for i in range(4))
            ):
                return p

            # diagonal izquierda
            if (
                r + 3 < ROWS
                and c - 3 >= 0
                and all(board[r+i, c-i] == p for i in range(4))
            ):
                return p

    return 0


def _get_value(board, player, V):

    tablero_perspectiva = board * player

    key = _state_key(tablero_perspectiva)

    return V.get(key, 0.0)


def _mejor_accion(board, player, V, cols_libres):

    mejor_valor = -np.inf

    mejor_col = cols_libres[
        len(cols_libres) // 2
    ]

    # priorizar centro
    cols_ordenadas = sorted(
        cols_libres,
        key=lambda c: abs(c - COLS // 2)
    )

    for col in cols_ordenadas:

        sucesor = _colocar(
            board,
            col,
            player
        )

        valor = _get_value(
            sucesor,
            player,
            V
        )

        if valor > mejor_valor:

            mejor_valor = valor
            mejor_col = col

    return mejor_col


def _td_update(path, ganador, V, lr, discount):

    reward = {
        1: 1.0,
        -1: -1.0
    }.get(ganador, 0.0)

    for board, player in reversed(path):

        state = board * player
        key = _state_key(state)

        v = V.get(key, 0.0)

        V[key] = v + lr * (reward - v)

        reward = discount * reward

def _recompensa(ganador, player):

    if ganador == player:
        return 1.0

    elif ganador == -player:
        return -1.0

    else:
        return 0.0