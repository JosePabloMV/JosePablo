import torch as pt
from collections import namedtuple, deque
SAVEFILE = "model"
LOADFILE = "test_model"
LOADMODE = True

INPUTS = 6
OUTPUTS = 3
class Breaker_Learning():

    class ReplayMemory(object):
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        def __init__(self, capacity):
            self.memory = deque([],maxlen=capacity)


        def push(self, *args):
            """Save a transition"""
            self.memory.append(self.Transition(*args))

        def sample(self, batch_size,random):
            transitions = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*transitions))

        def __len__(self):
            return len(self.memory)

    class Snake_brain(pt.nn.Module):
        def __init__(self):
            '''
            Se crea la red neuronal
            '''
            super().__init__()
            self.l1 = pt.nn.Sequential(
            pt.nn.Linear(INPUTS, 100),
            #pt.nn.LeakyReLU
            pt.nn.Tanh()
            )
            self.l2 = pt.nn.Sequential(
            pt.nn.Linear(100, 100),
            pt.nn.LeakyReLU()
            #pt.nn.LeakyReLU()
            )
            self.l3 = pt.nn.Sequential(
            pt.nn.Linear(100, OUTPUTS)
            )
            self.dropout = pt.nn.Dropout(0.10)

        def forward(self, x):
            x = self.l1(x)
            #x = self.dropout(x) 
            x = self.l2(x)
            #x = self.dropout(x)
            x = self.l3(x)
            return x
    
    def __init__(self,learning_rate,memory_capacity,batch_size,c_iters,discount_factor,load = LOADMODE):
        '''
        Main Variables
        '''

        if load:
            checkpoint = pt.load(LOADFILE)
            self.brain_target = self.Snake_brain()
            self.brain_policy = self.Snake_brain()
            self.brain_policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            # They have begin at the same time
            self.brain_target = self.Snake_brain()
            self.brain_policy = self.Snake_brain()
            # self.optimizer = pt.optim.RMSprop(self.brain_policy.parameters(), lr=learning_rate) 
        self.optimizer = pt.optim.Adam(self.brain_policy.parameters(), lr=learning_rate)	# Optimizador
        self.loss_fn = pt.nn.MSELoss()
        #self.loss_fn = pt.nn.SmoothL1Loss() # You can change it
        if load:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])	# Crea y carga el optimizador
        self.new_target_nn()
        '''
        Other Variables 
        '''
        self.epoch = 0
        self.batch_size = batch_size  
        self.c_iters = c_iters
        self.iterations = 0
        self.discount_factor = discount_factor
        self.memory = self.ReplayMemory(memory_capacity)
        self.memory_capacity = memory_capacity

    # Method that get the batch to train
    def get_batch(self,prng):
        if len(self.memory) < self.memory_capacity:
            return None
        return self.memory.sample(self.batch_size,prng)
    
    # Method that change the Target NN as Policy NN 
    def new_target_nn(self):
        self.brain_target.load_state_dict(self.brain_policy.state_dict())

    # Method to optimize the model
    def optimize(self,loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Juan_Problem: Look for this change , this is not natural, this can be a problem 
        # for param in self.brain_policy.parameters():
        #     param.grad.data.clamp_(-1, 1) # https://deepai.org/machine-learning-glossary-and-terms/gradient-clipping#:~:text=Gradient%20clipping%20is%20a%20technique,input%20into%20a%20specific%20output.
        self.optimizer.step()

    # Method that make the training in Policiy NN 
    def train(self,prng):
        # train  
        batch = self.get_batch(prng)
        if batch is None:
            return 
        #self.train(batch,current_apples)
        # Get other batch forms

        self.iterations +=1
        self.epoch+=1
        self.learn()
        non_final_mask = pt.tensor(tuple(map(lambda s: s is not None, # Lógica de estados finales 
                                          batch.next_state)),dtype=pt.bool)#,device=device) # Pone si es o no normal, ayuda en el reward
        non_final_next_states = pt.cat([s for s in batch.next_state
                                                if s is not None]) # Mete estados que no son finales
        state_batch = pt.cat(batch.state).reshape((self.batch_size,INPUTS)) # Le quita la tupla
        action_batch = pt.cat(batch.action).reshape(1,self.batch_size)
        reward_batch = pt.cat(batch.reward)
        state_action_values = self.brain_policy(state_batch).gather(1, action_batch) # Juan Gather makes the magic GatherBackward0
        next_state_values = pt.zeros(self.batch_size) #device=device) # Juan If is not state not update is made
        next_state_values[non_final_mask] = self.brain_target(non_final_next_states).max(1)[0].detach() # Chose better and detach

        # Compute the expected Q values.
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch # Ojo que aunque este en 0, el reward siempre juega
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(0)) # Esto esta raro que este 1, 50 TODO
        self.optimize(loss)
        self.save_model(loss.item())
        
    def learn(self):
        # ask for c_iters        
        if(self.iterations==self.c_iters):
            #change NN 
            self.new_target_nn()
            self.iterations=0


    # Do a single best movement TODO: to know if this is nn to use and if has to detach or not
    def best_movement(self,current_state):
        self.brain_policy.eval()
        with pt.no_grad():
            movement = self.brain_policy(pt.Tensor(current_state)).argmax().item()
        self.brain_policy.train()

        return movement

    # To set a new memory in the agent brain
    def new_memory(self,state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

       



    def save_model(self,loss):
        pt.save({
            'epoch': self.epoch,
            'model_state_dict': self.brain_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss#,
            #'current_apples': current_apples
            }, SAVEFILE)

    def save_model_perfect(self,manzanas):
        pt.save({
            'epoch': self.epoch,
            'model_state_dict': self.brain_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'loss': loss#,
            #'current_apples': current_apples
            }, "modelo_mejor_88")



