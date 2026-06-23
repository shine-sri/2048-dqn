import numpy as np
import random

class Game2048Env:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()

    def get_state(self):
        # One-hot encode the board states (16 layers deep for powers of 2 up to 2^16)
        state = np.zeros((16, self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] > 0:
                    power = int(np.log2(self.board[i, j]))
                    if power <= 16:
                        state[power - 1, i, j] = 1.0
        return state

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def slide_and_merge_row(self, row):
        non_zero = row[row > 0]
        new_row = []
        skip = False
        score_gain = 0
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                new_row.append(non_zero[i] * 2)
                score_gain += non_zero[i] * 2
                skip = True
            else:
                new_row.append(non_zero[i])
                
        new_row = np.array(new_row)
        return np.pad(new_row, (0, self.size - len(new_row)), 'constant'), score_gain

    def step(self, action):
        """
        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        Returns: next_state, reward, done
        """
        old_board = self.board.copy()
        score_gain = 0
        
        # Rotate board to standardize the processing array using LEFT swiping logic
        if action == 0:   # UP
            self.board = np.rot90(self.board, 1)
        elif action == 1: # DOWN
            self.board = np.rot90(self.board, -1)
        elif action == 3: # RIGHT
            self.board = np.rot90(self.board, 2)

        for i in range(self.size):
            self.board[i], gain = self.slide_and_merge_row(self.board[i])
            score_gain += gain

        # Rotate back to original orientation
        if action == 0:
            self.board = np.rot90(self.board, -1)
        elif action == 1:
            self.board = np.rot90(self.board, 1)
        elif action == 3:
            self.board = np.rot90(self.board, 2)

        moved = not np.array_equal(old_board, self.board)
        
        # --- FIX 1: THE INFINITE LOOP PREVENTER ---
        if not moved:
            return self.get_state(), -10.0, False
            
        # If the move was valid, add a new tile
        self.add_random_tile()
        
        # --- FIX 3: REWARD SHAPING (Strategic Math) ---
        # Base reward is the points scored from merging
        reward = float(score_gain)
        
        # 1. Empty Space Bonus: Force the AI to value a clean, uncluttered board
        empty_spaces = len(np.where(self.board == 0)[0])
        reward += empty_spaces * 2.0
        
        # 2. The Corner Heuristic: Punish the AI if it pulls the largest tile out of a corner
        max_tile = np.max(self.board)
        corners = [
            self.board[0, 0], self.board[0, self.size-1], 
            self.board[self.size-1, 0], self.board[self.size-1, self.size-1]
        ]
        
        if max_tile > 16 and max_tile not in corners:
            reward -= max_tile * 0.5 
            
        # --- FIX 2: THE 2048 WIN CONDITION ---
        if np.max(self.board) >= 2048:
            return self.get_state(), reward + 1000.0, True
            
        # --- Check for Standard Game Over ---
        done = True
        if 0 in self.board:
            done = False
        else:
            for i in range(self.size):
                for j in range(self.size - 1):
                    if self.board[i][j] == self.board[i][j+1]:
                        done = False
            for j in range(self.size):
                for i in range(self.size - 1):
                    if self.board[i][j] == self.board[i+1][j]:
                        done = False
                        
        return self.get_state(), reward, done

    # --- ACTION MASKING UPGRADE ---
    def get_available_moves(self):
        moves = []
        for action in range(4):
            old_board = self.board.copy()
            
            if action == 0:   self.board = np.rot90(self.board, 1)
            elif action == 1: self.board = np.rot90(self.board, -1)
            elif action == 3: self.board = np.rot90(self.board, 2)

            for i in range(self.size):
                self.board[i], _ = self.slide_and_merge_row(self.board[i])

            if action == 0:   self.board = np.rot90(self.board, -1)
            elif action == 1: self.board = np.rot90(self.board, 1)
            elif action == 3: self.board = np.rot90(self.board, 2)

            # If the board changed, the move is legal
            if not np.array_equal(old_board, self.board):
                moves.append(action)
                
            self.board = old_board.copy() # Reset for the next test
            
        return moves
