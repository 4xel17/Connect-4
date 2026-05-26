import sys
import os

# ============================================================
#  CONFIGURAR PATH DEL PROYECTO
# ============================================================

sys.path.insert(
    0,
    os.path.abspath(os.path.dirname(__file__))
)

import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy

# Importar el agente de Graph Search
from groups.GroupA.policy import MinimaxGraphSearchAgent


# ============================================================
#  RANDOM AGENT
#  (Baseline sin búsqueda)
# ============================================================

class RandomAgent(Policy):

    def mount(self):
        pass

    def act(self, n):
        """
        Selecciona una acción aleatoria.
        No realiza Graph Search.
        """

        actions = [a for a in range(7) if n[0, a] == 0]

        return int(np.random.choice(actions))


# ============================================================
#  ENTORNO DE JUEGO
# ============================================================

def play_game(policy_MIN, policy_MAX, verbose=False):
    """
    Ejecuta una partida completa de Connect 4.

    MIN_PLAYER = -1 (rojo)
    MAX_PLAYER =  1 (amarillo)
    """

    state = ConnectState()

    while not state.is_final():

        # ====================================================
        # TURNOS
        # ====================================================

        if state.player == -1:
            action = policy_MIN.act(state.board)

        else:
            action = policy_MAX.act(state.board)

        # ====================================================
        # VISUALIZACIÓN
        # ====================================================

        if verbose:
            print(f"Jugador {state.player} ejecuta acción a={action}")

        # ====================================================
        # ψ(s,e)
        # FUNCIÓN DE TRANSICIÓN
        # ====================================================

        state = state.transition(action)

    return state.get_winner()


# ============================================================
#  TEST 1
#  Minimax Graph Search vs Random
#  Minimax = MAX_PLAYER (amarillo)
# ============================================================

N_GAMES = 50

wins = 0
losses = 0
draws = 0

for _ in range(N_GAMES):

    graph_search_agent = MinimaxGraphSearchAgent()
    graph_search_agent.mount()

    random_agent = RandomAgent()
    random_agent.mount()

    # Random juega primero (MIN)
    # Graph Search juega segundo (MAX)

    result = play_game(
        random_agent,
        graph_search_agent
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    if result == 1:
        wins += 1

    elif result == -1:
        losses += 1

    else:
        draws += 1


print("\n====================================================")
print("TEST 1: GRAPH SEARCH AGENT vs RANDOM AGENT")
print("Minimax Graph Search = MAX_PLAYER (amarillo)")
print("Random Agent         = MIN_PLAYER (rojo)")
print("====================================================")

print(f"Partidas:   {N_GAMES}")
print(f"Victorias:  {wins} ({wins/N_GAMES*100:.0f}%)")
print(f"Derrotas:   {losses}")
print(f"Empates:    {draws}")


# ============================================================
#  TEST 2
#  Minimax Graph Search vs Random
#  Minimax = MIN_PLAYER (rojo)
# ============================================================

wins2 = 0
losses2 = 0
draws2 = 0

for _ in range(N_GAMES):

    graph_search_agent = MinimaxGraphSearchAgent()
    graph_search_agent.mount()

    random_agent = RandomAgent()
    random_agent.mount()

    # Graph Search juega primero (MIN)
    # Random juega segundo (MAX)

    result = play_game(
        graph_search_agent,
        random_agent
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    if result == -1:
        wins2 += 1

    elif result == 1:
        losses2 += 1

    else:
        draws2 += 1


print("\n====================================================")
print("TEST 2: GRAPH SEARCH AGENT vs RANDOM AGENT")
print("Minimax Graph Search = MIN_PLAYER (rojo)")
print("Random Agent         = MAX_PLAYER (amarillo)")
print("====================================================")

print(f"Partidas:   {N_GAMES}")
print(f"Victorias:  {wins2} ({wins2/N_GAMES*100:.0f}%)")
print(f"Derrotas:   {losses2}")
print(f"Empates:    {draws2}")