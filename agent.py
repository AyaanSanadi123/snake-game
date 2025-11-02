import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent :
    
    def __init__(self):
        # REMOVE all the resume_from_game logic
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.epsilon = 1.0 # Default starting epsilon
    def get_state(self,game):
        # this function converts complex game logic into a list of 11 items,
        # it answers 3 questions 1. where is the danger, 2.which way am i moving and 3. where is the food
        # here the function defines the 4 blocks around the snakes head 
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y) # One block left
        point_r = Point(head.x + 20, head.y) # One block right
        point_u = Point(head.x, head.y - 20) # One block up
        point_d = Point(head.x, head.y + 20) # One block down


        # here the code aims to get the snakes absolute current direction as a true or false value
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int)
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
       
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def save_checkpoint(self, record):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, 'checkpoint.pth')
        
        # Save all important state in one dictionary
        torch.save({
            'n_games': self.n_games,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'record': record,  # Save the high score
        }, file_name)
        print(f"Game {self.n_games}: Saving checkpoint...")

    def load_checkpoint(self):
        file_name = './model/checkpoint.pth'
        
        if os.path.exists(file_name):
            try:
                # Load the checkpoint
                checkpoint = torch.load(file_name)
                
                # Restore the state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.n_games = checkpoint['n_games']
                self.epsilon = checkpoint['epsilon']
                record = checkpoint.get('record', 0) # Get record, default to 0
                
                self.model.train() # Ensure model is in training mode
                
                print(f"Loaded checkpoint. Resuming from game {self.n_games + 1}. Record: {record}")
                return record # Return the saved record
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                return 0 # Start with 0 record
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0 # Start with 0 record
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    # 1. Create a blank agent
    agent = Agent()
    # 2. Load its progress and get the high score
    # This replaces the hardcoded resume logic
    record = agent.load_checkpoint() 

    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if score > record:
                record = score
                # Save the full checkpoint
                agent.save_checkpoint(record) 

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()