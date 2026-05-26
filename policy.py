import numpy as np
import math
from connect4.policy import Policy


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state.copy()
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0

    def ucb1(self, c=1.4):
        if self.N == 0:
            return float("inf")
        return (self.W / self.N) + c * math.sqrt(math.log(self.parent.N) / self.N)


class OhYes(Policy):

    def __init__(self):
        self.n_simulations = 120 

    def mount(self, action_timeout=None):
        self.n_simulations = max(100, int((action_timeout or 1.0) * 120))


    # UTILITIES


    def _available(self, s):
        return [c for c in range(7) if s[0, c] == 0]

    def _apply(self, s, col, player):
        ns = s.copy()
        for r in range(5, -1, -1):
            if ns[r, col] == 0:
                ns[r, col] = player
                break
        return ns

    def _check_win(self, s, player):
        for r in range(6):
            for c in range(4):
                if all(s[r, c+i] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(7):
                if all(s[r+i, c] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(4):
                if all(s[r+i, c+i] == player for i in range(4)):
                    return True

        for r in range(3, 6):
            for c in range(4):
                if all(s[r-i, c+i] == player for i in range(4)):
                    return True

        return False

    def _whose_turn(self, s):
        return 1 if np.sum(s == 1) <= np.sum(s == -1) else -1

    def _winning_move(self, s, player):
        for c in self._available(s):
            if self._check_win(self._apply(s, c, player), player):
                return c
        return None


    # SIMULATION LIGERA


    def _simulate(self, s, my_player):
        s = s.copy()
        player = my_player

        for _ in range(30):
            av = self._available(s)
            if not av:
                return 0.0

            # ganar inmediato
            for c in av:
                if self._check_win(self._apply(s, c, player), player):
                    return 1.0 if player == my_player else -1.0

            # random ligero
            col = int(np.random.choice(av))
            s = self._apply(s, col, player)

            if self._check_win(s, player):
                return 1.0 if player == my_player else -1.0

            player = -player

        return 0.0

  
    # BACKPROP
 

    def _backprop(self, node, reward):
        while node is not None:
            node.N += 1
            node.W += reward
            reward = -reward
            node = node.parent

    # ACT (FINAL POLICY)


    def act(self, s):

        my_player = self._whose_turn(s)
        opp = -my_player

        #  1. ganar inmediato
        win = self._winning_move(s, my_player)
        if win is not None:
            return win

        #  2. bloquear rival
        block = self._winning_move(s, opp)
        if block is not None:
            return block

   
        # MCTS
 
        root = MCTSNode(s)

        for _ in range(self.n_simulations):

            node = root

            # Selection
            while node.children:
                av = self._available(node.state)
                unexplored = [a for a in av if a not in node.children]

                if unexplored:
                    break

                node = max(node.children.values(), key=lambda n: n.ucb1())

            # Expansion
            av = self._available(node.state)
            if av:
                col = int(np.random.choice(av))
                player = self._whose_turn(node.state)
                new_state = self._apply(node.state, col, player)

                child = MCTSNode(new_state, node)
                node.children[col] = child
                node = child

            # Simulation
            reward = self._simulate(node.state, my_player)

            # Backprop
            self._backprop(node, reward)

        # mejor acción
        if not root.children:
            return int(np.random.choice(self._available(s)))

        return int(max(root.children, key=lambda c: root.children[c].N))
