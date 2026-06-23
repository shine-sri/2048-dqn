import time
import torch
import os
import numpy as np
from environment import Game2048Env
from agent import DQNAgent

def watch_ai():
    env = Game2048Env()
    agent = DQNAgent()
    
    if os.path.exists("2048_brain.pth"):
        checkpoint = torch.load("2048_brain.pth", weights_only=True)
        if 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded trained brain! (Trained for {checkpoint['episode']} total matches)")
        else:
            agent.model.load_state_dict(checkpoint)
            print("Successfully loaded trained brain!")
        
        # Set to 0 so it uses 100% of its learned math and doesn't guess randomly
        agent.epsilon = 0.0 
    else:
        print("No brain found. Playing randomly.")
        agent.epsilon = 1.0 
    
    state = env.reset()
    done = False
    move_count = 0
    move_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    print("\nStarting AI Exhibition Game...\n")
    
    while not done:
        move_count += 1
        print(f"--- Move {move_count} ---")
        for row in env.board:
            print("\t".join([str(val) if val > 0 else "." for val in row]))
        
        # --- THE SAFETY NET ---
        # Ask the neural network to rank all 4 moves mathematically
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.model(state_tensor)[0]
        
        # Sort actions from highest predicted score to lowest
        ranked_actions = torch.argsort(q_values, descending=True).tolist()
        
        valid_move_taken = False
        for action in ranked_actions:
            old_board = env.board.copy()
            
            # Try the move in the environment
            next_state, reward, step_done = env.step(action)
            
            # Did the board physically change?
            if not np.array_equal(old_board, env.board):
                print(f"\nAI swipes: {move_names[action]}\n")
                state = next_state
                done = step_done
                valid_move_taken = True
                break
        
        # If ALL 4 directions are blocked, the game is truly over
        if not valid_move_taken:
            print("\nNo valid moves left! True Game Over.")
            break
            
        time.sleep(0.5) 
        
    print("Game Over!")
    print("Final Board State:")
    for row in env.board:
        print("\t".join([str(val) if val > 0 else "." for val in row]))

if __name__ == "__main__":
    watch_ai()