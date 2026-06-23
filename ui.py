import pygame
import torch
import os
import time
import numpy as np
from environment import Game2048Env
from agent import DQNAgent

# --- EXACT 2048 COLOR SCHEME ---
COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
TEXT_COLORS = {
    "dark": (119, 110, 101),
    "light": (249, 246, 242)
}

def draw_board(screen, board, font):
    screen.fill((187, 173, 160)) # Background
    
    for i in range(4):
        for j in range(4):
            val = board[i][j]
            # Tiles larger than 2048 default to dark gray
            color = COLORS.get(val, (60, 58, 50)) 
            
            # Draw the tile
            rect = pygame.Rect(j * 100 + 10, i * 100 + 10, 80, 80)
            pygame.draw.rect(screen, color, rect, border_radius=5)
            
            # Draw the number
            if val > 0:
                text_color = TEXT_COLORS["dark"] if val <= 4 else TEXT_COLORS["light"]
                # Dynamically shrink font for huge numbers
                current_font = font if val < 1000 else pygame.font.SysFont("arial", 28, bold=True)
                text_surface = current_font.render(str(val), True, text_color)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)
                
    pygame.display.flip()

def play_gui():
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("PyTorch DQN Agent - 2048")
    font = pygame.font.SysFont("arial", 36, bold=True)
    
    env = Game2048Env()
    agent = DQNAgent()
    
    # Load the trained brain
    if os.path.exists("2048_brain.pth"):
        checkpoint = torch.load("2048_brain.pth", weights_only=True)
        if 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Brain loaded! Trained for {checkpoint['episode']} episodes.")
        else:
            agent.model.load_state_dict(checkpoint)
        agent.epsilon = 0.0 # 0% random guesses, 100% neural network
    else:
        print("No brain found! Ensure training has saved '2048_brain.pth'")
        return
    
    state = env.reset()
    done = False
    
    print("\nGraphical Exhibition Started! Check the new window.")
    
    while not done:
        # Allow user to close the window mid-game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
        draw_board(screen, env.board, font)
        time.sleep(0.15) # Adjust this to make the AI play faster or slower
        
        # Neural Network evaluates the board
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.model(state_tensor)[0]
            
        ranked_actions = torch.argsort(q_values, descending=True).tolist()
        
        valid_move_taken = False
        for action in ranked_actions:
            old_board = env.board.copy()
            next_state, reward, step_done = env.step(action)
            
            # Verify the move was legal
            if not np.array_equal(old_board, env.board):
                state = next_state
                done = step_done
                valid_move_taken = True
                break
                
        if not valid_move_taken:
            print("Game Over! No valid moves remaining.")
            break

    # Keep the final board on screen until the user physically closes the window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

if __name__ == "__main__":
    play_gui()