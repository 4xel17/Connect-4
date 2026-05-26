import numpy as np
from connect4.policy import Policy
from typing import override


class RandomAgent(Policy):

    @override
    def mount(self):
        pass

    @override
    def act(self, s):
        # columnas válidas
        valid_cols = [c for c in range(7) if s[0, c] == 0]

        # escoger una al azar
        return int(np.random.choice(valid_cols))