import numpy as np          # para operaciones matriciales sobre el tablero
from connect4.policy import Policy  # clase base que todo agente debe heredar
import random               # para exploracion aleatoria y simulaciones


class SaraFirstVisitMc(Policy):

    # Q-table compartida entre instancias: mapea (estado, accion) -> valor promedio
    # Se mantiene como variable de clase para persistir entre episodios
    q_table = {}

    # Contador de visitas por (estado, accion): necesario para calcular el promedio incremental
    # Primera visita MC requiere saber cuantas veces se visito cada par
    visit_count = {}

    def mount(self, *args, **kwargs):
        # mount() se llama una vez al iniciar el agente
        # Inicializa la memoria del episodio actual: lista de (estado, accion) visitados
        self.states_actions = []

    # ------------------------------------------------------------
    # 1. DETECCION DEL JUGADOR
    # ------------------------------------------------------------
    def get_player(self, s):
        # Cuenta fichas rojas (valor -1) en el tablero
        reds = np.sum(s == -1)
        # Cuenta fichas amarillas (valor 1) en el tablero
        yellows = np.sum(s == 1)

        # Si hay igual cantidad de fichas, es turno del jugador rojo (-1)
        # porque rojo siempre empieza primero
        if reds == yellows:
            return -1

        # Si rojo tiene mas fichas que amarillo, es turno de amarillo (1)
        return 1

    # ------------------------------------------------------------
    # 2. MOVIMIENTOS DISPONIBLES
    # ------------------------------------------------------------
    def available_moves(self, s):
        # Una columna esta disponible si su celda superior (fila 0) esta vacia (== 0)
        # Retorna lista de indices de columna jugables
        return [c for c in range(7) if s[0, c] == 0]

    # ------------------------------------------------------------
    # 3. SIMULACION DE JUGADAS
    # ------------------------------------------------------------
    def simulate_move(self, s, col, player):
        # Copia el tablero para no modificar el estado original
        board = s.copy()

        # Busca la fila mas baja disponible en la columna elegida (gravedad)
        for r in range(5, -1, -1):         # recorre de abajo (fila 5) hacia arriba (fila 0)
            if board[r, col] == 0:         # encuentra la primera celda vacia
                board[r, col] = player     # coloca la ficha del jugador
                break                      # sale del loop, solo ocupa una celda

        return board    # retorna el tablero resultante tras el movimiento

    # ------------------------------------------------------------
    # 4. DETECCION DE VICTORIA
    # ------------------------------------------------------------
    def is_winning_move(self, board, player):
        rows = 6    # numero de filas del tablero Connect-4
        cols = 7    # numero de columnas del tablero Connect-4

        # Verifica 4 en linea horizontal
        for r in range(rows):
            for c in range(cols - 3):      # hasta col 3 para que quepan 4 fichas
                if all(board[r, c + i] == player for i in range(4)):
                    return True            # encontro 4 fichas consecutivas en fila

        # Verifica 4 en linea vertical
        for r in range(rows - 3):          # hasta fila 2 para que quepan 4 fichas
            for c in range(cols):
                if all(board[r + i, c] == player for i in range(4)):
                    return True            # encontro 4 fichas consecutivas en columna

        # Verifica diagonal descendente (\)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return True            # encontro 4 fichas en diagonal \

        # Verifica diagonal ascendente (/)
        for r in range(3, rows):           # empieza en fila 3 para que quepan 4 hacia arriba
            for c in range(cols - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return True            # encontro 4 fichas en diagonal /

        return False    # ningun patron de victoria encontrado

    # ------------------------------------------------------------
    # 5. REPRESENTACION DEL ESTADO COMO LLAVE
    # ------------------------------------------------------------
    def board_to_key(self, s):
        # Convierte la matriz 6x7 en una tupla plana de 42 elementos
        # Las tuplas son hashables y pueden usarse como llave de diccionario
        return tuple(s.flatten())

    # ------------------------------------------------------------
    # 6. SIMULACION MONTE CARLO (rollout aleatorio hasta fin de juego)
    # ------------------------------------------------------------
    def rollout(self, s, player):
        # Simula una partida completa desde el estado s con movimientos aleatorios
        # Retorna: +1 si gana player, -1 si pierde, 0 si empate

        board = s.copy()        # copia del tablero para no modificar el original
        current = player        # empieza a jugar el jugador indicado

        for _ in range(42):     # maximo 42 turnos (6 filas x 7 columnas = 42 celdas)
            moves = self.available_moves(board)     # calcula movimientos disponibles

            if not moves:
                return 0        # tablero lleno sin ganador: empate

            col = random.choice(moves)              # elige columna aleatoriamente (rollout)
            board = self.simulate_move(board, col, current)  # aplica el movimiento

            if self.is_winning_move(board, current):
                # Si el jugador actual gano: retorna +1 si es nuestro jugador, -1 si es el rival
                return 1 if current == player else -1

            current = -current  # alterna el turno entre jugadores (-1 <-> 1)

        return 0    # se agotaron los 42 turnos sin ganador: empate

    # ------------------------------------------------------------
    # 7. ACTUALIZACION DE Q-TABLE (Monte Carlo First Visit)
    # ------------------------------------------------------------
    def update_q_table(self, reward):
        # Actualiza los valores Q usando el metodo Monte Carlo First Visit
        # Solo actualiza la PRIMERA vez que se visito cada par (estado, accion) en el episodio

        visited = set()     # conjunto para rastrear que pares ya fueron actualizados

        for state_key, action in self.states_actions:
            pair = (state_key, action)      # par (estado, accion) a evaluar

            if pair in visited:
                # Si ya fue visitado en este episodio, IGNORAR (First Visit MC)
                # Solo la primera visita al par en el episodio contribuye al promedio
                continue

            visited.add(pair)   # marcar este par como visitado en el episodio actual

            # Obtener contador de visitas anteriores (0 si nunca se visito)
            n = self.visit_count.get(pair, 0)

            # Obtener valor Q actual (0 si es la primera vez)
            q = self.q_table.get(pair, 0.0)

            # Incrementar contador de visitas para este par
            self.visit_count[pair] = n + 1

            # Actualizar Q con promedio incremental: Q_nuevo = Q_viejo + (reward - Q_viejo) / (n+1)
            # Esta formula calcula el promedio acumulado sin guardar todos los retornos anteriores
            self.q_table[pair] = q + (reward - q) / (n + 1)

        # Limpiar el historial del episodio para el siguiente turno
        self.states_actions = []

    # ------------------------------------------------------------
    # 8. FUNCION PRINCIPAL DE DECISION
    # ------------------------------------------------------------
    def act(self, s: np.ndarray) -> int:
        # Seguridad: inicializa states_actions si mount() no fue llamado
        if not hasattr(self, "states_actions"):
            self.states_actions = []

        player = self.get_player(s)     # identifica de quien es el turno
        opponent = -player              # calcula el jugador contrario
        moves = self.available_moves(s) # lista de columnas jugables

        # --- a. MOVIMIENTO GANADOR INMEDIATO ---
        for move in moves:
            next_board = self.simulate_move(s, move, player)   # simula colocar ficha
            if self.is_winning_move(next_board, player):
                # Si este movimiento nos da la victoria inmediata, jugarlo ya
                state_key = self.board_to_key(s)
                self.states_actions.append((state_key, move))
                self.update_q_table(1.0)    # recompensa positiva: ganamos
                return int(move)

        # --- b. BLOQUEO DE VICTORIA DEL OPONENTE ---
        for move in moves:
            next_board = self.simulate_move(s, move, opponent)  # simula jugada del rival
            if self.is_winning_move(next_board, opponent):
                # Si el oponente ganaria en ese movimiento, bloquearlo
                state_key = self.board_to_key(s)
                self.states_actions.append((state_key, move))
                self.update_q_table(0.5)    # recompensa moderada: evitamos perder
                return int(move)

        # --- c. SELECCION POR MONTE CARLO FIRST VISIT ---
        state_key = self.board_to_key(s)    # convierte tablero actual en llave

        best_move = None            # mejor movimiento encontrado
        best_value = -float('inf')  # peor valor posible como punto de partida

        num_rollouts = 5            # numero de simulaciones aleatorias por movimiento candidato

        for move in moves:
            pair = (state_key, move)    # par (estado actual, accion candidata)

            # Obtener valor Q aprendido previamente (0.0 si nunca se visito)
            q_value = self.q_table.get(pair, 0.0)

            # Realizar multiples rollouts para estimar el valor de este movimiento
            rollout_results = []
            for _ in range(num_rollouts):
                # Simula colocar la ficha en esta columna
                next_board = self.simulate_move(s, move, player)

                if self.is_winning_move(next_board, player):
                    # Si la simulacion detecta victoria inmediata, valor maximo
                    rollout_results.append(1.0)
                else:
                    # Continua la simulacion con movimientos aleatorios hasta el final
                    result = self.rollout(next_board, player)
                    rollout_results.append(result)

            # Promedio de los resultados de los rollouts para este movimiento
            avg_rollout = sum(rollout_results) / len(rollout_results)

            # Bonus estrategico por cercania al centro (columnas centrales son mejores)
            # La columna 3 recibe bonus 0, columnas adyacentes menos, extremos el menor bonus
            center_bonus = -abs(3 - move) * 0.05

            # Valor total = promedio ponderado entre Q aprendido y resultado del rollout
            # 0.6 * rollout + 0.4 * Q: da mas peso al rollout actual que al historico
            total_value = 0.6 * avg_rollout + 0.4 * q_value + center_bonus

            # Actualizar mejor movimiento si este tiene mayor valor total
            if total_value > best_value:
                best_value = total_value
                best_move = move

        # --- d. EXPLORACION EPSILON-GREEDY ---
        # Con 10% de probabilidad, elige un movimiento aleatorio
        # Esto garantiza exploracion de estados no visitados (necesario para MC)
        if random.random() < 0.1:
            best_move = random.choice(moves)    # movimiento completamente aleatorio

        # Registrar el par (estado, accion) elegido en el historial del episodio
        self.states_actions.append((state_key, best_move))

        # Simular el resultado del movimiento elegido para actualizar la Q-table
        next_board = self.simulate_move(s, best_move, player)   # aplica el movimiento
        reward = self.rollout(next_board, player)               # estima la recompensa final

        # Actualizar Q-table con el retorno estimado (Monte Carlo First Visit)
        self.update_q_table(reward)

        return int(best_move)   # retorna la columna elegida como entero