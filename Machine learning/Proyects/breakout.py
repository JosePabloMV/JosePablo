# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import collections
import enum
from hashlib import new
from pickle import TRUE
import numpy as np
from PIL import Image,ImageTk
import random
import time
import torch
from ML import Breaker_Learning 
try:
    import tkinter as tk
    from tkinter import simpledialog
except ImportError:
    try:
        import Tkinter as tk
    except ImportError:
        print("Missing library: Tkinter, please install")

DISSABLE_BLOCKS = False
RANDOM_SEED = 21
SNAKE_SEED = 31337
HEIGHT = 40#10
WIDTH = 60#20
INIT_LENGTH = 6
INIT_MOVES = 256
INIT_DIRECTION = (35,30)#(8,10)


MEMORY_CAPACITY = 5 # Es un escenario pequeño deberia de aprender rapido 
BATCH_SIZE = 5 # Caso similar al anterior
CITER=5
LEARNING_RATE = 0.001# Esto puede ser [0.1,0.01,0.001]
DISCOUNT = 0.9  # Esto puede ser [0.1,0.01,0.001]
GREDDY = 0.50 # Puede subirle mas, pero la idea es que una vez es aleaotorio y otras no
MINGREDDY = 0.80 # Por que no quiere moverse
DECAY = 0.00001 # Por que no quiere moverse (si lo aumenta seguiria subiendo greedy, tenga cuidado)


### TODO: CUSTOMIZE THE REWARDS
EMPTY_REWARD = -100
FOLLOW_REWARD = 100
BLOCK_REWARD = 1000
DEAD_REWARD = -1000
#SELF_REWARD = 0

### UI related  
FPS = 60
APS = 20

colors = {
    0: (0,0,0),
    1: (255,255,128),
    2: (0,255,0),
    3: (255,0,0),
    8: (255,0,0),
    10: (255,110,0),
    12: (255,255,0),
    14: (30,180,30),
    16: (0,210,255),
    18: (55,55,200),
}

### Q LEARNING VARIABLES ###
AC = "action"
EP = "EpsilonGreddy"
DE = "Decay" 

class Agent():
    # Initializes the training model
    # Input states for the model depend on the get_state method, which can be modified
    def __init__(self, memory_capacity, batch_size, c_iters, learning_rate, discount_factor, eps_greedy, decay):
        self.prng = random.Random()
        self.prng.seed(RANDOM_SEED)
         # Initialize the two main vectors
        self.STATS =  dict()
        self.STATS[AC] = {0:0,1:1,2:3} 
        self.STATS[EP] = eps_greedy  
        self.STATS[DE] = decay
        self.prng = random.Random()
        self.prng.seed(RANDOM_SEED)
        self.brain = Breaker_Learning(learning_rate,memory_capacity,batch_size,c_iters,discount_factor)
    
    # Performs a complete simulation by the agent, then samples "batch" memories from memory and learns, afterwards updates the decay
    def simulation(self, env):
        # Reset enviroment to begin a new simulation
        env.reset()
        # Changes made by decay just affect eps_greddy
        eps_greddy = self.STATS[EP]
        # Do simulation
        simulation_flag = False
        # Movement 
        current_movement = 0

        while not simulation_flag:
            self.step(env) # step_tuple = ((reward, state),action)  
            # Change the new eps_greddy
            self.STATS[EP] *= (1+self.STATS[DE])
            # Ask for the end game
            simulation_flag = env.is_terminal_state() 
            current_movement+=1
        self.STATS[EP] = eps_greddy
        print(current_movement)
    
    # Performs a single step of the simulation by the agent, if learn=False memories are not stored
    def step(self, env, learn=True):
        # Get current state
        state = env.get_state() #
        random_movement = False
        movement = 0
        # if learn=False no updates are performed and the best action is always taken
        if learn:
            eps_greddy = self.prng.random()
            if eps_greddy > self.STATS[EP]:
                random_movement = True
        # if random action happen, choose random action
        if random_movement:
            movement = self.prng.randint(0,2) 
        else:
            movement = self.brain.best_movement(state)
        # Perform the action selected
        action =  self.STATS[AC][movement]
        env_tuple = env.perform_action(action) 
        reward = env_tuple[0] # TODO : FALTA DEFINIR LOS PREMIOS
        if learn:
            if env.is_terminal_state():
                next_state = None
            else:
                next_state = torch.tensor([env_tuple[1]],dtype = torch.float32) # TODO : REVISAR ESE ENV_TUPLE, ESTABA ENCERRADO EN []
            reward = torch.tensor([reward])
            action =  torch.tensor([movement])
            state = torch.tensor([state],dtype = torch.float32) # voy
            self.brain.new_memory(state, action, next_state, reward)
            self.brain.train(self.prng)

        else:
            print(action)

class Breakout():
    def __init__(self, seed=SNAKE_SEED):
        self.prng = random.Random()
        self.seed = seed
        self.reset()
    
    def get_score(self):
        return self.score

    # Resets the environment
    def reset(self):
        self.prng.seed(self.seed)
        self.board = np.zeros((HEIGHT, WIDTH))
        self.valid_positions = set((i,j) for i in range(HEIGHT) for j in range(WIDTH))
        # Ball State
        self.ball = INIT_DIRECTION
        self.ball_direction = (-1,  1 if self.prng.random() < 0.5 else -1 )
        self.paddle = collections.deque([])
        self.board[self.ball] = 3
        for i in range(INIT_LENGTH):
            self.paddle.append((self.ball[0]+1, self.ball[1]-2+i))
            self.board[self.paddle[-1]] = 1
        self.moves_left = INIT_MOVES # TODO : To define how many blocks are
        self.dead = False
        self.score = 0

        if DISSABLE_BLOCKS :
            self.blocks =[]
        else:
            self.blocks = self.BlocksGrid(8)
            for i in self.blocks.grid:
                for j in i.block:
                    self.board[j]= j[0] if j[0]%2 == 0 else j[0]-1
    
    def get_state(self):
        # state =  torch.zeros((INIT_LENGTH),dtype=torch.float32)
        state = [-1 for _ in range(INIT_LENGTH)]
        for index in range (INIT_LENGTH):
            if self.paddle[index][1] == self.ball[1]:
                state[index] = 1
                break
        return state
    
    # Returns whether current state is terminal
    def is_terminal_state(self):
        return self.dead or self.blocks.blocks_cleared() or self.moves_left == 0
    
    def change_ball(self):
        scored_point = False
        ball_new_position = (self.ball[0] + self.ball_direction[0], self.ball[1]+self.ball_direction[1])
        new_direction = list(self.ball_direction)
        # Block TODO: Jopa
        # print(ball_new_position)

        free_block = True
        if (self.ball[0] + self.ball_direction[0],self.ball[1]) in self.blocks: # Direct contact from bottom or top side
            free_block = False
            new_direction[0] = -new_direction[0]
            self.board = self.blocks.delete_block((self.ball[0] + self.ball_direction[0],self.ball[1]),self.board)
            self.score += 1
            scored_point = True
        
        if (self.ball[0],self.ball[1]+ self.ball_direction[1]) in self.blocks: # Direct contact from left or right side
            free_block = False
            new_direction[1] = -new_direction[1]
            self.board = self.blocks.delete_block((self.ball[0] ,self.ball[1] + self.ball_direction[1]),self.board)
            self.score += 1
            scored_point = True

        if free_block and ball_new_position in self.blocks: # Direct contact comes from diagonal
            self.board = self.blocks.delete_block(ball_new_position,self.board)
            # THIS CAN BE CHANGED TODO
            new_direction[0] = -new_direction[0]
            self.score += 1
            scored_point = True
        
        if scored_point:
            self.moves_left = INIT_MOVES
        else: 
            self.moves_left -= 1

    # TODO : Decirle que el juego terminó
        # Wall
        wall = 0
        if(ball_new_position[0]<0): # Top Wall
            new_direction[0] = 1

        if(ball_new_position[1]<0): # Left Wall
            new_direction[1] = 1
            wall = 1

        elif(ball_new_position[1] == WIDTH): # Right Wall
            new_direction[1] = -1
            wall = -1
        
        ball_new_position = (ball_new_position[0],ball_new_position[1]+wall)
        # # Paddle
        if ball_new_position in self.paddle:
            if wall==0:
               new_direction[1] = -1 if self.paddle.index(ball_new_position) < 3 else 1
            new_direction[0] = -1
        
        # Finish Game
        if ball_new_position[0] == HEIGHT:
            self.dead = True

        # Change Ball
        else:
            new_ball = (self.ball[0] + new_direction[0] , self.ball[1] + new_direction[1] )

            self.board[new_ball] = 3
            self.board[self.ball] = 0
            self.ball = new_ball
            self.ball_direction = tuple(new_direction)
        return scored_point
    def move_paddle_left(self):
        if self.paddle[0][1] != 0:
            delete_square = self.paddle.pop()
            new_square = (self.paddle[0][0],self.paddle[0][1]-1)
            self.paddle.appendleft(new_square)
            self.board[new_square] = 1
            self.board[delete_square] = 0

    def move_paddle_right(self):
        if self.paddle[-1][1] != WIDTH-1:
            delete_square = self.paddle.popleft()
            new_square = (self.paddle[-1][0],self.paddle[-1][1]+1)
            self.paddle.append(new_square)
            self.board[new_square] = 1
            self.board[delete_square] = 0
    
    # Performs an action and returns its reward and the new state
    def perform_action(self, action):
        # 4: ←← 3: ←, 2: _, 1: →, 0: →→
        # Paddle movement
        # print(action)
        if action == 0:            
            for i in range(3): self.move_paddle_right()
        elif action == 1:
            for i in range(2): self.move_paddle_right()
        elif action == 3:
            for i in range(2): self.move_paddle_left()
        elif action == 4:
            for i in range(3): self.move_paddle_left()
        
        current_points = self.score
        scored_point = self.change_ball()
        
        state = self.get_state() 
        reward = EMPTY_REWARD
        if 1 in state:
            reward = FOLLOW_REWARD
        if scored_point:
            reward = BLOCK_REWARD * (self.score-current_points)
            if self.score != 1: print(reward,self.score)
        if self.dead:
            reward = DEAD_REWARD
        return reward,state


    class BlocksGrid():
        def __init__(self,block_size):
            self.grid = collections.deque([])
            self.generate_grid(block_size)
            
        def generate_grid(self,block_size):
            num_col = block_size//2
            y = np.linspace(0,WIDTH-num_col,WIDTH//num_col).astype(int)
            x = np.array(range(8,20,2))
            for i in x:
                for j in y:
                    self.grid.append(self.Block((i,j),block_size))

        def delete_block(self, coor, board):
            to_delete = ()
            for block in self.grid:
                if coor in block:
                    to_delete = block
                    break
            if to_delete: 
                self.grid.remove(to_delete)
                for block in to_delete.block:
                    board[block] = 0
            return board
        
        def blocks_cleared(self):
            return len(self.grid) == 0

        def __contains__(self, key):
            for block in self.grid:
                if key in block:
                    return True
            return False
            
        class Block():
            def __init__(self,corner_1,block_size):
                self.block = collections.deque([])
                self.generate_block(corner_1,block_size)
            
            def generate_block(self, corner_1,block_size):
                num_col = block_size//2
                for i in range(block_size):
                    self.block.append((corner_1[0]+(i//num_col),corner_1[1]+(i%num_col)))
            
            def __contains__(self, key):
                return key in self.block
                

        
from threading import Timer

class mainWindow():
    def __init__(self,agentCls=Agent):
        self.breakout = Breakout()
        self.agentCls = agentCls
        # Control
        self.redraw = False
        self.playing_user = False
        self.playing_agent = False
        self.last_direction = 0
        self.last_action = 0
        self.epoch = 0
        self.score = 0
        self.highscore = 0
        self.memory_capacity = MEMORY_CAPACITY#10000
        self.batch_size = BATCH_SIZE#1000
        self.c_iters = CITER#10
        self.learning_rate = LEARNING_RATE#0.001
        self.discount = DISCOUNT#0.25
        self.greedy = GREDDY#0.25
        self.decay = DECAY#1e-7
        self.agent = agentCls(self.memory_capacity, self.batch_size, self.c_iters, self.learning_rate, self.discount, self.greedy, self.decay)
        # Interface
        self.root = tk.Tk()
        self.root.title("Snake AI")
        self.root.bind("<Configure>",self.resizing_event)
        self.root.bind("<Key>",self.key_press_event)
        self.frame = tk.Frame(self.root, width=650, height=550)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=1,height=1)
        # Control buttons
        self.buttonReset = tk.Button(self.frame, text="Reset",command=self.buttonReset_press,bg="indian red")
        self.buttonStep = tk.Button(self.frame, text="Step",command=self.buttonStep_press,bg="sea green")
        self.buttonSkip = tk.Button(self.frame, text="Skip",command=self.buttonSkip_press,bg="sea green")
        self.buttonPlayAgent = tk.Button(self.frame, text="Simulation",command=self.buttonPlayagent_press,bg="forest green")
        self.buttonPlayUser = tk.Button(self.frame, text="Play",command=self.buttonPlayuser_press,bg="forest green")
        self.epochString = tk.StringVar(value="Episodes: 0")
        self.scoreString = tk.StringVar(value="Score/Best: 0/0")
        self.epochLabel = tk.Label(self.frame,textvariable=self.epochString, relief=tk.RIDGE, padx=5, pady=2)
        self.scoreLabel = tk.Label(self.frame,textvariable=self.scoreString, relief=tk.RIDGE, padx=5, pady=2)
        self.mlLabel = tk.Label(self.frame,text="ML Controls:", relief=tk.RIDGE, padx=5, pady=2)
        self.humanLabel = tk.Label(self.frame,text="Entertainment for humans:", relief=tk.RIDGE, padx=5, pady=2)
        # Customization
        self.labelCustomization = tk.Label(self.frame, text="Customization", relief=tk.RIDGE, padx=5, pady=2)
        # Memory capacity, Batch size, Alpha learning rate, Gamma discount, Epsilon greedy
        self.stringMemorycap = tk.StringVar(value="Mem. capacity: "+str(self.memory_capacity))
        self.labelMemorycap = tk.Label(self.frame,textvariable=self.stringMemorycap, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonMemorycap = tk.Button(self.frame, text="Set", command=self.buttonMemorycap_press, bg="sea green")
        self.stringBatchsize = tk.StringVar(value="Batch size: "+str(self.batch_size))
        self.labelBatchsize = tk.Label(self.frame,textvariable=self.stringBatchsize, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonBatchsize = tk.Button(self.frame, text="Set", command=self.buttonBatchsize_press, bg="sea green")
        self.stringCiters = tk.StringVar(value="C-iters: "+str(self.c_iters))
        self.labelCiters = tk.Label(self.frame,textvariable=self.stringCiters, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonCiters = tk.Button(self.frame, text="Set", command=self.buttonCiters_press, bg="sea green")
        self.stringAlphalr = tk.StringVar(value="α-learning: "+str(self.learning_rate))
        self.labelAlphalr = tk.Label(self.frame,textvariable=self.stringAlphalr, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonAlphalr = tk.Button(self.frame, text="Set", command=self.buttonAlphalr_press, bg="sea green")
        self.stringGammadisc = tk.StringVar(value="γ-discount: "+str(self.discount))
        self.labelGammadisc = tk.Label(self.frame,textvariable=self.stringGammadisc, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonGammadisc = tk.Button(self.frame, text="Set", command=self.buttonGammadisc_press, bg="sea green")
        self.stringEpsilongreedy = tk.StringVar(value="ε-greedy: "+str(self.greedy))
        self.labelEpsilongreedy = tk.Label(self.frame,textvariable=self.stringEpsilongreedy, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEpsilongreedy = tk.Button(self.frame, text="Set", command=self.buttonEpsilongreedy_press, bg="sea green")
        self.stringEtadecay = tk.StringVar(value="η-decay: "+str(self.decay))
        self.labelEtadecay = tk.Label(self.frame,textvariable=self.stringEtadecay, relief=tk.RIDGE, padx=5, pady=2)
        self.buttonEtadecay = tk.Button(self.frame, text="Set", command=self.buttonEtadecay_press, bg="sea green")
        # Start
        self.root.after(0,self.update_loop)
        self.root.mainloop()
    
    # Resizing event
    def resizing_event(self,event):
        if event.widget == self.root:
            self.redraw = True
            self.canvas_width = max(event.width - 250,1)
            self.canvas_height = max(event.height - 40,1)
            self.frame.configure(width=event.width,height=event.height)
            self.canvas.configure(width=self.canvas_width,height=self.canvas_height)
            self.canvas.place(x=20,y=20)
            if (WIDTH/HEIGHT)*self.canvas_height > self.canvas_width:
                self.board_width,self.board_height = self.canvas_width,int((HEIGHT/WIDTH)*self.canvas_width)
            else:
                self.board_height,self.board_width = self.canvas_height,int((WIDTH/HEIGHT)*self.canvas_height)
            self.board_offset_x,self.board_offset_y = (self.canvas_width - self.board_width)//2,(self.canvas_height - self.board_height)//2
            self.mlLabel.place(x=event.width - 210, y=20, width=200)
            self.epochLabel.place(x=event.width - 210, y=50)
            self.scoreLabel.place(x=event.width - 210, y=80)
            self.buttonReset.place(x=event.width - 195, y=110, width=50)
            self.buttonStep.place(x=event.width - 130, y=110, width=50)
            self.buttonSkip.place(x=event.width - 65, y=110, width=50)
            self.buttonPlayAgent.place(x=event.width - 155, y=145, width=100)
            self.humanLabel.place(x=event.width - 210, y=200, width=200)
            self.buttonPlayUser.place(x=event.width - 155, y=235, width=100)
            # Customization
            self.labelCustomization.place(x=event.width - 210, y=290, width=190)
            self.labelMemorycap.place(x=event.width - 210, y=320)
            self.buttonMemorycap.place(x=event.width - 50, y=320)
            self.labelBatchsize.place(x=event.width - 210, y=350)
            self.buttonBatchsize.place(x=event.width - 50, y=350)
            self.labelCiters.place(x=event.width - 210, y=380)
            self.buttonCiters.place(x=event.width - 50, y=380)
            self.labelAlphalr.place(x=event.width - 210, y=410)
            self.buttonAlphalr.place(x=event.width - 50, y=410)
            self.labelGammadisc.place(x=event.width - 210, y=440)
            self.buttonGammadisc.place(x=event.width - 50, y=440)
            self.labelEpsilongreedy.place(x=event.width - 210, y=470)
            self.buttonEpsilongreedy.place(x=event.width - 50, y=470)
            self.labelEtadecay.place(x=event.width - 210, y=500)
            self.buttonEtadecay.place(x=event.width - 50, y=500)
    
    # Key press event
    def key_press_event(self,event):
        direction = (event.keycode - 38)%4
        if self.direction == 3:
            if direction == 3:
                self.direction = 4
            else:
                self.direction = direction
        elif self.direction == 1:
            if direction == 1:
                self.direction = 0
            else:
                self.direction = direction
        else:
            self.direction = direction
        # print(self.direction)
    
    # Update loop
    def update_loop(self):
        if time.time() - self.last_action >= 1/APS:
            if self.playing_user and self.direction>=0:
                self.redraw = True
                self.last_direction = 2
                self.last_action = time.time()
                self.breakout.perform_action(self.direction)
                if self.breakout.is_terminal_state(): self.buttonPlayuser_press()
            elif self.playing_agent:
                self.redraw = True
                self.last_action = time.time()
                if not self.breakout.is_terminal_state(): self.agent.step(self.breakout, learn=False)

            self.direction = 2
        if self.redraw:
            self.redraw_canvas()
        self.root.after(int(1000/FPS),self.update_loop)


    def reset_action(self):
        self.direction = 2
    # Play User button
    def buttonPlayuser_press(self):
        if not self.playing_agent:
            self.playing_user = not self.playing_user
            self.buttonPlayUser.configure(text=("Stop" if self.playing_user else "Play"), bg=("indian red" if self.playing_user else "forest green"))
            if self.playing_user:
                self.direction = -1
                self.last_direction = -10
                self.last_action = 0
                self.breakout = Breakout(seed=int(time.time()*1000))
                self.redraw = True
    
    # Resets the learning model
    def buttonReset_press(self):
        if self.playing_agent or self.playing_user: return
        self.breakout = Breakout()
        self.agent = self.agentCls(self.memory_capacity, self.batch_size, self.c_iters, self.learning_rate, self.discount, self.greedy, self.decay)
        self.epoch = 0
        self.score,self.highscore = 0,0
        self.epochString.set("Episodes: %i" % (self.epoch,))
        self.scoreString.set("Score/Best: %i/%i" % (self.score,self.highscore))
        self.redraw = True
    
    # Executes an epoch
    def buttonStep_press(self):
        if self.playing_agent or self.playing_user: return
        self.epoch += 1
        self.breakout = Breakout()
        self.agent.simulation(self.breakout)
        self.score = self.breakout.get_score()
        self.highscore = max(self.highscore, self.score)
        self.epochString.set("Episodes: %i" % (self.epoch,))
        self.scoreString.set("Score/Best: %i/%i" % (self.score,self.highscore))
        self.redraw = True
    
    # Executes epochs until epoch X
    def buttonSkip_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askinteger("Run simulations", "How many simulations:", parent=self.root, minvalue=1, initialvalue=10)
        if x:
            for i in range(x):
                self.buttonStep_press()
                
    # Play Agent button
    def buttonPlayagent_press(self):
        if not self.playing_user:
            self.breakout = Breakout()
            self.playing_agent = not self.playing_agent                
            self.buttonPlayAgent.configure(text=("Stop" if self.playing_agent else "Simulation"), bg=("indian red" if self.playing_agent else "forest green"))
            self.redraw = True
    
    # Memory-cap button
    def buttonMemorycap_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askinteger("Memory capacity", "Input the memory capacity:", parent=self.root,minvalue=1)
        if x:
            self.memory_capacity = x
            self.stringMemorycap.set("Mem. capacity: "+str(self.memory_capacity))
            self.buttonReset_press()
    
    # Batch-size button
    def buttonBatchsize_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askinteger("Memory capacity", "Input the batch size:", parent=self.root,minvalue=1)
        if x:
            self.batch_size = x
            self.stringBatchsize.set("Batch size: "+str(self.batch_size))
            self.buttonReset_press()
    
    # C-iters button
    def buttonCiters_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askinteger("C-iterations", "Input the c-iterations:", parent=self.root,minvalue=1)
        if x:
            self.c_iters = x
            self.stringCiters.set("C-iters: "+str(self.c_iters))
            self.buttonReset_press()
    
    # Alpha-lr button
    def buttonAlphalr_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askfloat("α Learning rate", "Input the learning rate:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.learning_rate = x
            self.stringAlphalr.set("α-learning: "+str(self.learning_rate))
            self.buttonReset_press()
    
    # Gamma-disc button
    def buttonGammadisc_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askfloat("γ Discount factor", "Input the discount factor:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.discount = x
            self.stringGammadisc.set("γ-discount: "+str(self.discount))
            self.buttonReset_press()
    
    # Epsilon-greedy button
    def buttonEpsilongreedy_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askfloat("ε Greedy", "Input the initial ε greedy value:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.greedy = x
            self.stringEpsilongreedy.set("ε-greedy: "+str(self.greedy))
            self.buttonReset_press()
    
    # Eta-decay button
    def buttonEtadecay_press(self):
        if self.playing_agent or self.playing_user: return
        x = tk.simpledialog.askfloat("η Decay factor", "Input the η decay factor for ε:", parent=self.root,minvalue=0,maxvalue=1)
        if x:
            self.decay = x
            self.stringEtadecay.set("η-decay: "+str(self.decay))
            self.buttonReset_press()
    
    def redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0,0,self.canvas_width,self.canvas_height,fill="#606060",width=0)
        pixels = np.array( [[colors[y] for y in x] for x in self.breakout.board] )
        self.image = Image.fromarray(pixels.astype('uint8'), 'RGB')
        self.photo = ImageTk.PhotoImage(image=self.image.resize((self.board_width,self.board_height),resample=Image.NEAREST))
        self.canvas.create_image(self.board_offset_x,self.board_offset_y,image=self.photo,anchor=tk.NW)
        dy = self.board_height / HEIGHT
        dx = self.board_width / WIDTH
        for i in range(1,HEIGHT):
            self.canvas.create_line(self.board_offset_x, self.board_offset_y+int(dy*i), self.board_offset_x+self.board_width,self.board_offset_y+int(dy*i))
        for i in range(1,WIDTH):
            self.canvas.create_line(self.board_offset_x + int(dx*i), self.board_offset_y, self.board_offset_x+int(dx*i),self.board_offset_y+self.board_height)
        self.canvas.create_rectangle(self.board_offset_x,self.board_offset_y,self.board_offset_x+self.board_width,self.board_offset_y+self.board_height,outline="#0000FF",width=3)
        self.redraw = False

if __name__ == "__main__":
    x = mainWindow()
