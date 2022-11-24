import numpy as np


class Board:

    def __init__(
            self,
            size: int,
            seed: int):
        """
        IN:
            - size: the size of the square board
            - seed: the seed for the random generator
        """
        self.size = size
        self.current_state = np.zeros(
            shape=(size, size),
            dtype=int
            )
        self.previous_state = np.zeros(
            shape=(size, size),
            dtype=int
            )
        self.generator = np.random.default_rng(seed=seed)
        self.current_score = 0
        self.previous_score = 0
        self.add_elements(n=2)

    def get_random_empty_position(self) -> tuple[int, int]:
        """
        Return an index of a random empty cell in the current boar
        in the form of a tuple
        """
        zero_indexes = np.where(self.current_state == 0)
        k = self.generator.integers(len(zero_indexes[0]))
        return zero_indexes[0][k], zero_indexes[1][k]

    def save_state(self) -> None:
        """
        Save the current state, which is the board and the score.
        """
        self.previous_state[:] = self.current_state
        self.previous_score = self.current_score

    def restore_state(self) -> None:
        """
        Restore the previous state
        """
        self.current_state[:] = self.previous_state
        self.current_score = self.previous_score
    
    def move_horizontal(self, right: bool) -> int:
        """
        Perform the move of the game in a horizontal way.
        IN:
            - right: True for moving right, False for moving left
        OUT:
            - the number of compacted cell
        """
        compact = Board.compact_right if right else Board.compact_left
        cell_removed = 0
        for i in range(self.size):
            row = self.current_state[i]
            non_zeros, c_score, c_cell_removed = compact(row[row!=0])
            self.current_score += c_score
            cell_removed += c_cell_removed
            zeros = np.zeros(
                shape=(self.size-len(non_zeros),),
                dtype=int
                )
            self.current_state[i] = np.concatenate(
                (zeros, non_zeros, ) if right else
                (non_zeros, zeros, )
            )        
        return cell_removed

    def move_vertical(self, up: bool) -> int:
        """
        Perform the move of the game in a horizontal way.
        IN:
            - up: True for moving up, False for moving down
        OUT:
            - the number of compacted cell
        """
        self.current_state = self.current_state.T
        cell_removed = self.move_horizontal(right=(not up))
        self.current_state = self.current_state.T
        return cell_removed

    def move(self, direction: str) -> None:
        self.save_state()
        if direction == 'up':
            self.move_vertical(up=True)
        elif direction == 'down':
            self.move_vertical(up=False)
        elif direction == 'right':
            self.move_horizontal(right=True)
        elif direction == 'left':
            self.move_horizontal(right=False)
        else:
            raise Exception(f'Direction {direction} is not valid.')
        self.add_elements(n=1)

    def add_elements(self, n: int) -> None:
        for _ in range(n):
            u = self.generator.uniform()
            value = 2 if u<0.8 else 4
            self.current_state[self.get_random_empty_position()] = value

    @staticmethod
    def compact_right(a: list) -> tuple[list, int, int]:
        score = 0
        n_cell_removed = 0
        start = 0
        end = len(a)-1
        i = start
        while i < end:
            if a[i] == a[i+1]:
                a[i+1] += a[i]
                score += a[i+1]
                n_cell_removed += 1
                j = i
                while j>start:
                    a[j] = a[j-1]
                    j -= 1
                start += 1
                i += 1
            i += 1
        return a[start:], score, n_cell_removed
        
    @staticmethod
    def compact_left(a: list) -> tuple[list, int, int]:
        score = 0
        start = 0
        n_cell_removed = 0
        end = len(a) - 1
        i = end
        while i > start:
            if a[i] == a[i-1]:
                a[i-1] += a[i]
                score += a[i-1]
                n_cell_removed += 1
                j = i
                while j < end-1:
                    a[j] = a[j+1]
                    j += 1
                end -= 1
                i -= 1
            i -= 1
        end += 1
        return a[:end+1], score, n_cell_removed
