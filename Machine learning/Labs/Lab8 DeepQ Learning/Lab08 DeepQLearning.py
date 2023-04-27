# -*- coding: utf-8 -*-
"""
@author: Blopa

Estudiante: Jose Pablo Mora Villalobos
Carné: B85326

Para este laboratorio, primero intenté con una arquitectura convolucional para el estado que retornaba el 
incialmente el ambiente, con una imagen del tablero en tres canales. La red consitía en dos capas convolucionales
que redcuían los datos a un tensor de 16 valores después de ser aplanados, y una capa densa que recivía las 16 
entradas y tenía solo las 4 neuronas de salida. Las recompensas eran:

EMPTY_REWARD = 0
APPLE_REWARD = 1000
WALL_REWARD = -100
SELF_REWARD = -100

Estas recompensas se mantuvieron por casi todas las pruebas que hice con otras arquitecturas y estados. El problema
con esta red era que tardaba mucho en entrenarse y cuando realizé pruebas tenía un error con el entrenamiento de la 
red que no me había dado cuenta y por lo tanto esta red no convergió.
Realizé tra prueba con una red convolucionl similar, pero el estado era una sola imagen con la misma información que 
estába en los tres canales antiores, de esta forma el cuerpo de la serpierte tenía el número 1, la cabeza 2 y la manzana 3.
Para esta prueba ya había solucionado el error y fue la primera que empezó a dar resultados buenos, o sea que convergía
pero tardaba mucho entrenándose y en ciertas ocaciones se perdía la convergencia y no podía encontrar ninguna manzana.

Casi paralelamente a la primera red intenté con una red densa, la cual se encuentra en este código. Tiene tres capas densas, la primera 
recibe 8 valores del estado y los convierte a 16, la segunda capa concecta estos valores con otras 16 neuronas y la capa 
de salida recibe los 16 valores y da los 4 para cada acción. El estado en este caso está compuesto por un tensor de 8 valores
donde los primeros 2 son las coordenadas de la manzana en el tablero, los siguientes 2 son las coordenas de la cabeza de la
serpiente y los últimos 4 son un 1 si el siguiente cuadro en esa dirección es un muro o el cuerpo de la serpiete y un 0 si no.
Estos úlitmos 4 valores siguen la convención de 0 arriba, 1 derecha, 2 abajo y 3 izquierda. 

Con esta red intenté una combinación de recompensas para el agente. Hice pruebas dándole en vez de una recompensa vacía, una
proporcianal a la distacia en que estaba de la manzana después de realizar un moviento, entonces alejarse de la manzana le daba
menos puntos que si se acercaba. Intenté con 2 tipos de distacia, la manhattan  y la euclidiana. Con ambas dio buenos reseultados
, pero  también los intenté con una recompensa vaciá y de igual forma dio buenos sirvió. Las demás recompensas son las misma 
que está ahora en el código.

También intenté entrenar a la red con baches grandes y una memoria grande, de 20000 y 50000 respectivamente. Esto dio buenos resultados
ya que la serpiente encontraba bastantes manzanas pero no era consistente. Después de unas 1000 iteraciones lograba encontrar 3 o 4
manzanas, y cada 1000 entrenamientos más, la serpiente recolectaba hasta 13 o se mantenía en 4 o 5. Por úlitmo, intenté con baches
pequeños de 10 y una memoria de 20. En este caso el entrenamiento toma mucho menos tiempo, entonces pude entrenar de 10000
en 10000 iteraciones. En este caso el resultado fue similar, después de 70000 iteraciones la serpiente logra 4 manzanas, pero puede
seguirse entrenando. Creo que con un batha más grande sería mejor, pero no tan grande como el 20000.

Con respecto a los hiperparámetros, el único que pude notar que tenía un peso grande en el resultado del entrenamiento era el greedy.
Con un greede bajo, no aprendía tan bien a encontrar las primeras manzana porque no lograba alcanzarlas consistentemente y usualmente 
tomaba el mismo camino (el mejor que sabe pues el greedy es bajo), entonces no aprendía tampoco las recompensas de llegar a ciertas 
casillas. En cambio con un greede de 0.5, no tan alto, lograba un buen balance entre exploración y aprovechamiento.

Por último, la información del ajente y de la red se guardan en un archivo llamado target_policy_data_local cada 100 iteraciones 
(sobrescribe el archivo existente) y se pueden cargar con el botón Load "Target-Policy" de la interfaz. 

Además, cuando se hace una simulación con el botón "Simulation" se guarda un gif con el recorrido. 

Se adjuntan un modelo con la última arquitectura descrita que se entrenó con los baches de 10 y la memoria de 20 por 70000 iteraciones, 
y también un gif con una simulación del modelo que se entrenó por 13000 iteraciones con un bache de 17500 y una memoria de 25000.
"""

import collections
import enum
from matplotlib.cbook import flatten
import numpy as np
from PIL import Image,ImageTk, ImageGrab
import random
import time
import torch
import tkinter.simpledialog
import time
from math import dist
from scipy.spatial.distance import cityblock
from collections import deque
import os

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    try:
        import Tkinter as tk
    except ImportError:
        print("Missing library: Tkinter, please install")

RANDOM_SEED = 21
SNAKE_SEED = 31337
WIDTH = 16
HEIGHT = 24
INIT_LENGTH = 3
INIT_MOVES = 256
INIT_DIRECTION = 0
EXTRA_MOVES = 128

### TODO: CUSTOMIZE THE REWARDS
EMPTY_REWARD = 0
APPLE_REWARD = 1000
WALL_REWARD = -100
SELF_REWARD = -100

### UI related[ -42.4186, -586.3904, -832.4617,  -79.4160]
FPS = 24
APS = 8

colors = {
    0: (0,0,0),
    1: (255,255,128),
    2: (0,255,0),
    3: (255,0,0)
}

### TODO: CUSTOMIZE AGENT

class TargetPolicyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(8,16),
            torch.nn.LeakyReLU(),
        )
        self.l2 = torch.nn.Sequential(
            torch.nn.Linear(16,16),
            torch.nn.LeakyReLU(),
        )
        self.l3 = torch.nn.Sequential(
            torch.nn.Linear(16,4),
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)

class Agent():
    # Initializes the training model
    # Input states for the model depend on the get_state method, which can be modified
    def __init__(self, memory_capacity, batch_size, c_iters, learning_rate, discount_factor, eps_greedy, decay):
        self.prng = random.Random()
        self.prng.seed(RANDOM_SEED)
        self.batch_size = batch_size
        self.c_iters = c_iters
        self.memory_capacity = memory_capacity
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps_greedy = eps_greedy
        self.decay = decay
        self.__target = TargetPolicyNN()
        self.__policy = TargetPolicyNN()
        self.__policy_op  = torch.optim.Adam(self.__policy.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_capacity)
        self.current_epoch = 0
        self.current_step = 0
        self.__processing_time = 0
        self.changes = 0
        
        pass
    
    # Performs a complete simulation by the agent, then samples "batch" memories from memory and learns, afterwards updates the decay
    def simulation(self, env):
        #pass
        # TODO: Implement simulation loop + learning loop

        while len(self.memory) != self.memory.maxlen:
            self.__fill_mem(env)
        self.current_epoch += 1
        env.reset()
        start = time.process_time_ns()
        # if self.current_epoch%50 == 0:
        #     print('Iter: {:>7.2f} - Time: {:>10.2f} - Greedy: {:.2f} - Changes: {}'.format(self.current_epoch, self.__processing_time,self.eps_greedy,self.changes))
        #     with torch.no_grad():
        #         pol = self.__policy(env.get_state().unsqueeze(0)).detach()
        #         tar = self.__target(env.get_state().unsqueeze(0)).detach()
        #         print('State: ',env.get_state())
        #         print('Policy: ', torch.argmax(pol,dim=1).item(), np.array(pol)[0])
        #         print('Target: ', torch.argmax(tar,dim=1).item() ,np.array(tar)[0])
            # self.changes = 0
        if self.current_epoch%100 == 0:
            self.__save_dnq()

        while not env.is_terminal_state():
            self.step(env)
            self.eps_greedy *= 1-self.decay
        self.__processing_time +=( (time.process_time_ns()-start)*1e-9)

    def __fill_mem(self, env):
        env.reset()
        while not env.is_terminal_state():
            self.step(env)


    # Performs a single step of the simulation by the agent, if learn=False memories are not stored
    def step(self, env, learn=True):
        
        # pass
        # TODO: Implement single step, if learn=False no updates are performed and the best action is always taken
        action_rand = random.random() #Compracaión con el greedy
        
        state = env.get_state()         #Estado actual
        
        if action_rand < max(0.10,self.eps_greedy) and learn:
            action = random.choice(env.get_actions())
        else:
            with torch.no_grad():
                q_nn_move = self.__policy(state.unsqueeze(0)).detach()
            action = torch.argmax(q_nn_move,dim=1).item()
            # if not learn:

            #     print('\nState: ',state)
            #     print('Actions: ',q_nn_move)
            #     print('Action: ',torch.argmax(q_nn_move,dim=1).item())
        reward, transition = env.perform_action(action) # R(s) y T(s,a)
        
        if learn:
            self.memory.append((state, action, reward, transition, env.is_terminal_state()))
            if len(self.memory) == self.memory.maxlen:
                if self.current_step % 1 == 0:
                    self.__train_policy(random.sample(self.memory,min(self.batch_size,len(self.memory))))
                if self.current_step % self.c_iters == 0:
                    self.__target.load_state_dict(self.__policy.state_dict())
                    self.changes+=1
                self.current_step += 1

    
    def __train_policy(self,x):
        loss_fn = torch.nn.MSELoss()
        
        with torch.no_grad():
            y_ = self.__target(torch.stack(self.__get_colum(x,3))).detach()
        self.__policy_op.zero_grad()                                   # Coloca los Δw en 0
        y_pred = self.__policy(torch.stack(self.__get_colum(x,0)))	   # Predice los valores del conjunto de entrenamiento
        y  = self.__calc_y(y_,x,torch.clone(y_pred).detach())
        loss = loss_fn(y_pred,y)		                               # Calcula la pérdida
        loss.backward()				                                   # Calcula el backprogration (Δw) y acumula el error
        self.__policy_op.step()					                       # Aplica los Δw acumulados y avanza un paso la iter.
    

    def __calc_y(self, y, x, y_pred):
        for sample,i in zip(x,range(len(x))):
            # (state, action, reward, transition, env.is_terminal_state())
            if sample[4]: #final state
                y_pred[i][sample[1]] = sample[2]
            else:
                y_pred[i][sample[1]] = sample[2] + self.discount_factor * torch.max(y[i]).item()
        return y_pred

    def __get_colum(self, data, col: int):
        return [row[col] for row in data]
    
    def load_dnq(self,filename):
        checkpoint = torch.load(filename,map_location=torch.device('cpu'))
        self.__policy.load_state_dict(checkpoint['policy_state_dict'])
        self.__policy_op.load_state_dict(checkpoint['optimizer_state_dict'])
        self.__target.load_state_dict(checkpoint['target_state_dict'])
        self.prng = (checkpoint['params'][0])
        self.prng.seed(RANDOM_SEED)
        self.memory_capacity = checkpoint['params'][1]
        self.batch_size = checkpoint['params'][2]
        self.c_iters = checkpoint['params'][3]
        self.learning_rate = checkpoint['params'][4]
        self.discount_factor = checkpoint['params'][5]
        self.eps_greedy = checkpoint['params'][6]
        self.decay = checkpoint['params'][7]
        self.current_epoch = checkpoint['epoch']
        print('New params: ', checkpoint['params'])

    def __save_dnq(self):        
        torch.save({ 
            'epoch': self.current_epoch,
            'policy_state_dict': self.__policy.state_dict(),
            'optimizer_state_dict': self.__policy_op.state_dict(),
            'target_state_dict': self.__target.state_dict(),

            'params':[  self.prng,
                        self.memory.maxlen,
                        self.batch_size,
                        self.c_iters, 
                        self.learning_rate,
                        self.discount_factor,
                        self.eps_greedy,
                        self.decay,
                        ],
            }, "target_policy_data_local")


class Action(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake():
    def __init__(self, seed=SNAKE_SEED):
        self.prng = random.Random()
        self.prng.seed(seed)
        self.seed = seed
        self.reset()
    
    def get_score(self):
        return self.score

    # Resets the environment
    def reset(self):
        random.seed(self.seed)
        self.prng.seed(self.seed)
        self.board = np.zeros((HEIGHT, WIDTH))
        self.valid_positions = set((i,j) for i in range(HEIGHT) for j in range(WIDTH))
        self.positions = collections.deque([])
        for i in range(INIT_LENGTH):
            self.positions.append(((HEIGHT//2)+i, WIDTH//2))
            self.board[self.positions[-1]] = 1 + int(i==0)
        self.direction = 0 # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        self.moves_left = INIT_MOVES
        self.dead = False
        self.score = 0
        self.__add_apple()
    
    def __add_apple(self):
        options = list(self.valid_positions - set(self.positions))
        if not options: return False
        apple = self.prng.choice(options)
        self.board[apple] = 3
        self.apple_position = apple
        return True

    # Returns action list            
    def get_actions(self):
        return [a.value for a in Action]

    ### TODO(OPTIONAL): CAN BE MODIFIED; CURRENTLY RETURNS TENSOR OF SHAPE (3, WIDTH, HEIGHT)
    
    def get_state(self):
        head = self.positions[0]
        num_moves = np.zeros(4) # UP, RIGHT, DOWN, LEFT
        for pos in list(self.positions)[1:]:
            if pos[0] == head[0]:
                if head[1] == pos[1]-1 or head[1] == WIDTH-1:
                    num_moves[1] = 1
                elif head[1] == pos[1]+1 or head[1] == 1:
                    num_moves[3] = 1
            if pos[1] == head[1]:
                if head[0] == pos[0]-1 or head[1] == HEIGHT-1:
                    num_moves[2] = 1
                elif head[0] == pos[0]+1 or head[1] == 1:
                    num_moves[0] = 1

        state = np.array([self.apple_position, head]).flatten()
        return torch.tensor(np.array([state,num_moves]).flatten(),dtype=torch.float32)
    
    # Returns whether current state is terminal
    def is_terminal_state(self):
        return self.dead or self.moves_left<=0

    # Performs an action and returns its reward and the new state
    def perform_action(self, action):
        reward = EMPTY_REWARD
        if abs(self.direction - action)!=2: self.direction = action
        tail = self.positions.pop()
        self.board[tail] = 0
        if self.direction%2 == 0:
            head = (self.positions[0][0] + (-1 if self.direction==0 else 1), self.positions[0][1])
            if head[0]<0 or head[0]>=HEIGHT:
                self.dead = True
                reward = WALL_REWARD
        else:
            head = (self.positions[0][0], self.positions[0][1] + (-1 if self.direction==3 else 1))
            if head[1]<0 or head[1]>=WIDTH:
                self.dead = True
                reward = WALL_REWARD
        if not self.dead:
            self.board[self.positions[0]] = 1
            self.positions.appendleft(head)
            value = self.board[head]
            self.board[head] = 2
            if value<3 and value>0:
                self.dead = True
                reward = SELF_REWARD
            elif value==3:
                self.board[tail] = 1
                self.positions.append(tail)
                if not self.__add_apple(): self.dead = True
                self.score += 1
                reward = APPLE_REWARD
        return reward,self.get_state()

class mainWindow():
    def __init__(self,agentCls=Agent):
        self.snake = Snake()
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
        self.memory_capacity = 10000
        self.batch_size = 1000
        self.c_iters = 10
        self.learning_rate = 0.001
        self.discount = 0.25
        self.greedy = 0.25
        self.decay = 1e-7
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
        # Jose Pablo: Added button to load policy and target
        self.buttonLoadDQN = tk.Button(self.frame, text="Load Target-Policy",command=self.buttonLoadDQN_press,bg="forest green")
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
        
        # Jose Pablo: Added a gif of the simulation
        self.gif = []

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
            self.buttonLoadDQN.place(x=event.width - 155, y=170, width=100)
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
    
    # Jose Pablo: Added a method to save a gif of the simulation
    def save_capture(self):
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        
        im = ImageGrab.grab((x0, y0, x1, y1))
        self.gif.append(im)
        
        #im.save('mypic.png') # Can also say im.show() to display it

    # Key press event
    def key_press_event(self,event):
        self.direction = (event.keycode - 38)%4
    
    # Update loop
    def update_loop(self):
        if time.time() - self.last_action >= 1/APS:
            if self.playing_user and self.direction>=0:
                self.redraw = True
                self.last_direction = self.direction
                self.last_action = time.time()
                self.snake.perform_action(self.direction)
                if self.snake.is_terminal_state(): self.buttonPlayuser_press()
            elif self.playing_agent:
                self.redraw = True
                self.last_action = time.time()
                if not self.snake.is_terminal_state(): self.agent.step(self.snake, learn=False)
                self.save_capture()
        if self.redraw:
            self.redraw_canvas()
        self.root.after(int(1000/FPS),self.update_loop)
    
    # Play User button
    def buttonPlayuser_press(self):
        if not self.playing_agent:
            self.playing_user = not self.playing_user
            self.buttonPlayUser.configure(text=("Stop" if self.playing_user else "Play"), bg=("indian red" if self.playing_user else "forest green"))
            if self.playing_user:
                self.direction = -1
                self.last_direction = -10
                self.last_action = 0
                self.snake = Snake(seed=int(time.time()*1000))
                self.redraw = True
    
    # Resets the learning model
    def buttonReset_press(self):
        if self.playing_agent or self.playing_user: return
        self.snake = Snake()
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
        self.snake = Snake()
        self.agent.simulation(self.snake)
        self.score = self.snake.get_score()
        self.highscore = max(self.highscore, self.score)
        self.epochString.set("Episodes: %i" % (self.epoch,))
        self.scoreString.set("Score/Best: %i/%i" % (self.score,self.highscore))
        self.redraw = True
    
    # Executes epochs until epoch X
    def buttonSkip_press(self):
        if self.playing_agent or self.playing_user: return
        # Cambié el initiavalue a 1000
        x = tkinter.simpledialog.askinteger("Run simulations", "How many simulations:", parent=self.root, minvalue=1, initialvalue=1000)
        if x:
            for i in range(x):
                self.buttonStep_press()
                
    # Play Agent button
    def buttonPlayagent_press(self):
        if self.playing_agent:            
            self.gif[0].save(f'simulation_{self.epoch}.gif',save_all=True, append_images=self.gif[1:], loop=0, duration=150)
        self.gif = []
        if not self.playing_user:
            self.snake = Snake()
            self.playing_agent = not self.playing_agent                
            self.buttonPlayAgent.configure(text=("Stop" if self.playing_agent else "Simulation"), bg=("indian red" if self.playing_agent else "forest green"))
            self.redraw = True
        
    
    # Jose Pablo: Load policy target data button
    def buttonLoadDQN_press(self):
        if not self.playing_user:
            self.snake = Snake()
            # filename = tkinter.simpledialog.askstring("File name", "Enter filename:", parent=self.root, initialvalue='target_policy_data_local')
            filename = tkinter.filedialog.askopenfilename(
                    title='Open a file',
                    initialdir=os.getcwd())
            if not filename: return
            self.agent.load_dnq(filename)
            self.epoch = self.agent.current_epoch
            self.epochString.set("Episodes: %i" % (self.epoch,))
            self.stringMemorycap.set("Mem. capacity: "+str(self.agent.memory_capacity))
            self.stringBatchsize.set("Batch size: "+str(self.agent.batch_size))
            self.stringCiters.set("C-iters: "+str(self.agent.c_iters))
            self.stringAlphalr.set("α-learning: "+str(self.agent.learning_rate))
            self.stringGammadisc.set("γ-discount: "+str(self.agent.discount_factor))
            self.stringEpsilongreedy.set("ε-greedy: "+str(self.agent.eps_greedy)[:5])
            self.stringEtadecay.set("η-decay: "+str(self.agent.decay))

            #self.playing_agent = not self.playing_agent                
            #self.buttonPlayAgent.configure(text=("Stop" if self.playing_agent else "Simulation"), bg=("indian red" if self.playing_agent else "forest green"))
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
        pixels = np.array( [[colors[y] for y in x] for x in self.snake.board] )
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
