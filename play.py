#this file is to play with the trained models 

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point # [cite: 2]
from model import Linear_QNet, QTrainer
from helper import plot # [cite: 3]
import os

MAX_MEMORY = 100_000
# BATCH_SIZE and LR are not needed for playing
# LR = 0.001
# BATCH_SIZE = 1000


class Agent :
    
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9 # Not strictly needed, but harmless
        self.memory = deque(maxlen=MAX_MEMORY) # Not strictly needed, but harmless
        self.model = Linear_QNet(11, 256, 3)
        
        # --- REMOVED ---
        # We don't need a trainer
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # We don't need epsilon (randomness)
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.999
        # self.epsilon = 1.0
        # --- --------- ---

    def get_state(self, game):
        # This function is unchanged
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]
        return np.array(state, dtype=int)
    
    # --- REMOVED remember, train_long_memory, and train_short_memory ---

    def get_action(self, state):
        # This function is now 100% exploitation (no randomness)
        final_move = [0,0,0]
        
        # --- REMOVED all epsilon/random logic ---
        # if random.random() < self.epsilon:
        #    ...
        # else:
        
        # AI always uses its "brain"
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move
    
    # --- REMOVED save_checkpoint ---

    def load_checkpoint(self):
        # This function is modified to only load the model weights
        file_name = './model/checkpoint.pth'
        
        if os.path.exists(file_name):
            try:
                checkpoint = torch.load(file_name) # 
                
                # Restore the model's "brain"
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # --- REMOVED optimizer, n_games, and epsilon loading ---
                # We can still load n_games and record for the print message
                self.n_games = checkpoint['n_games']
                record = checkpoint.get('record', 0)
                
                # --- IMPORTANT ---
                # Set model to "evaluation mode"
                self.model.eval() 
                
                print(f"Loaded checkpoint. Watching AI play. Record: {record}")
                return record # Return the saved record
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Cannot play.")
                return 0
        else:
            print("No checkpoint found. Cannot play.")
            return 0
    

def play_game(): # Renamed from train()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    agent = Agent()
    # Load the trained model
    record = agent.load_checkpoint() 

    game = SnakeGameAI() # [cite: 2]
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        # --- OPTIMIZATION ---
        # We wrap play_step in torch.no_grad()
        # This tells PyTorch not to calculate gradients, making it faster.
        with torch.no_grad():
            reward, done, score = game.play_step(final_move)
        
        # --- REMOVED training and remembering ---
        # state_new = agent.get_state(game)
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            
            # --- REMOVED all training, decay, and saving logic ---
            # agent.train_long_memory()
            # if agent.epsilon > agent.epsilon_min:
            #     ...
            # if score > record:
            #     ...
            #     agent.save_checkpoint(record) 

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # We can still plot the results [cite: 3]
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    play_game() # Call the renamed function