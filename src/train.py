import torch
from environment import Game2048Env
from agent import DQNAgent
import numpy as np

def train_engine(episodes=1500):
    env = Game2048Env()
    agent = DQNAgent()
    target_update_frequency = 10
    
    print("Beginning Training Loop... Neural Network is now exploring environment.")
    
    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        
        while True:
            # --- ACTION MASKING INTERCEPT ---
            available_moves = env.get_available_moves()
            if not available_moves:
                break # Board is completely full, game over
                
            action = agent.act(state, available_moves)
            # --------------------------------
            
            next_state, reward, done = env.step(action)
            
            # Store transition data into experience buffer
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            agent.replay()
            
            if done:
                break
                
        if e % target_update_frequency == 0:
            agent.update_target_network()
            
        if e % 50 == 0:
            max_tile = np.max(env.board)
            print(f"Episode: {e}/{episodes} | Max Tile Realized: {max_tile} | Epsilon: {agent.epsilon:.4f}")

            checkpoint = {
                'episode': e,
                'model_state_dict': agent.model.state_dict()
            }
            torch.save(checkpoint, "2048_brain.pth")

if __name__ == "__main__":
    train_engine()
