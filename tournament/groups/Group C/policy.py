import numpy as np
import math
import random
from connect4.policy import Policy


class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state.copy()
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
        self.untried = [c for c in range(7) if state[0, c] == 0]
        self.N = 0
        self.W = 0.0


class OhYes(Policy):

    def __init__(self):
        self.n_simulations = 180

    def mount(self, action_timeout=None):
        self.n_simulations = max(120, int((action_timeout or 1.0) * 150))

    # -------------------------
    # UTILITIES
    # -------------------------

    def _available(self, s):
        return [c for c in range(7) if s[0, c] == 0]

    def _apply(self, s, col, player):
        ns = s.copy()
        for r in range(5, -1, -1):
            if ns[r, col] == 0:
                ns[r, col] = player
                return ns
        return ns

    def _check_win(self, s, p):
        for r in range(6):
            for c in range(4):
                if all(s[r, c+i] == p for i in range(4)):
                    return True

        for r in range(3):
            for c in range(7):
                if all(s[r+i, c] == p for i in range(4)):
                    return True

        for r in range(3):
            for c in range(4):
                if all(s[r+i, c+i] == p for i in range(4)):
                    return True

        for r in range(3, 6):
            for c in range(4):
                if all(s[r-i, c+i] == p for i in range(4)):
                    return True

        return False

    # -------------------------
    # ROLLOUT (PURE RANDOM)
    # -------------------------

    def _rollout(self, s, player):
        for _ in range(30):
            av = self._available(s)
            if not av:
                return 0

            col = random.choice(av)
            s = self._apply(s, col, player)

            if self._check_win(s, player):
                return player

            player = -player

        return 0

    # -------------------------
    # UCB
    # -------------------------

    def _ucb(self, node, c=1.4):
        if node.N == 0:
            return float("inf")
        return (node.W / node.N) + c * math.sqrt(math.log(node.parent.N) / node.N)

    # -------------------------
    # SELECTION
    # -------------------------

    def _select(self, node):
        while not node.untried and node.children:
            node = max(node.children.values(), key=lambda n: self._ucb(n))
        return node

    # -------------------------
    # EXPANSION
    # -------------------------

    def _expand(self, node):
        col = node.untried.pop()
        new_state = self._apply(node.state, col, node.player)
        next_player = -node.player

        child = MCTSNode(new_state, next_player, node, col)
        node.children[col] = child
        return child

    # -------------------------
    # BACKPROP
    # -------------------------

    def _backprop(self, node, result, root_player):
        while node:
            node.N += 1
            if result == root_player:
                node.W += 1
            elif result != 0:
                node.W -= 1
            node = node.parent

    # -------------------------
    # ACTION
    # -------------------------

    def act(self, s):
        root_player = 1 if np.sum(s == 1) <= np.sum(s == -1) else -1
        root = MCTSNode(s, root_player)

        for _ in range(self.n_simulations):

            node = self._select(root)

            if node.untried:
                node = self._expand(node)

            result = self._rollout(node.state, node.player)

            self._backprop(node, result, root_player)

        if not root.children:
            return int(np.random.choice(self._available(s)))

        return max(root.children, key=lambda c: root.children[c].N)