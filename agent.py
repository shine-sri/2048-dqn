import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from model import DQN2048

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=20000) # Memory bank capacity
        self.gamma = 0.99                 # Discount factor for future rewards
        self.epsilon = 1.0                # Initial exploration probability
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.batch_size = 64
        
        self.model = DQN2048()
        self.target_model = DQN2048()     # Used to stabilize target calculations
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # --- UPGRADED ACT FUNCTION ---
    def act(self, state, available_moves=None):
        # Fallback if no specific moves are provided
        if available_moves is None or len(available_moves) == 0:
            available_moves = [0, 1, 2, 3]

        # Epsilon-Greedy: Pick a RANDOM valid move
        if random.random() <= self.epsilon:
            return random.choice(available_moves)
        
        # Neural Network: Pick the BEST valid move
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)[0]
            
        best_move = available_moves[0]
        max_q = -float('inf')
        
        # Filter the math to only consider legal swipes
        for move in available_moves:
            if q_values[move].item() > max_q:
                max_q = q_values[move].item()
                best_move = move
                
        return best_move

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in mini_batch]))
        actions = torch.LongTensor([x[1] for x in mini_batch]).unsqueeze(1)
        rewards = torch.FloatTensor([x[2] for x in mini_batch])
        next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch]))
        dones = torch.FloatTensor([float(x[4]) for x in mini_batch])

        # Current predicted values
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Max future expected values computed via stable target network (Bellman Equation)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay