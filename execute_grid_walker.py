import gym
import grid_walker
from collections import deque
import numpy as np
import time
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Conv2D, LSTM, Reshape, Dense, Dropout, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD

import keras

# Classe de memória, para salvar as experiência observadas
class Memory():
    def __init__(self, max_size=2000):
        self.states = deque(maxlen=max_size)
        self.targets = deque(maxlen=max_size)

	# Adiciona um vetor à memória
    def add(self, state, target):
        self.states.append(state)
        self.targets.append(target)

	# Retorna 'batch_size' amostras aleatórias
    def sample(self, batch_size):
        size = min(batch_size, len(self.states))
        idx = np.random.choice(np.arange(len(self.states)), size=size, replace=False)
        states = [self.states[i] for i in idx]
        targets = [self.targets[i] for i in idx]
        return [np.array(states), np.array(targets)]

model_channels = 2
grid_size = 5
def create_reward_model():
    global model_channels, grid_size
    action_input = Input(shape=(grid_size,grid_size,model_channels), name='action_input') 

    x = Conv2D(128, (3,3), activation='relu', padding='same')(action_input)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    x = Concatenate()([x,Flatten()(action_input)])

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dense(4, activation='linear')(x)
    model = Model(inputs=action_input, outputs=x)
    model.compile(Adam(lr=5e-5),loss='mse')
    return model

""" def create_reward_model_multiple_input():
    input1 = Input(shape=(3,3,1), name='input1') 
    input1 = Input(shape=(3,3,1), name='input1') 
    x = action_input


    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu')(x)
    model = Model(inputs=action_input, outputs=x)
    model.compile(Adam(lr=1e-4),loss='mse')
    return model
 """
    


env = grid_walker.GridWalker(model_channels) 
render = True

model = create_reward_model()
memory = Memory()

try:
    model.load_weights('./model_weights.h5')
    print('Weights loaded')
except:
    print('No model weights found')


gamma = 0.9		       # Fator de esquecimento
eps = 0  		       # Fator de exploração inicial
decay_factor = 1       # Fator de decaimento de eps
num_episodes = 100000     # Número de simulações
r_list = []
lt = time.time()
for i in range(num_episodes):
    h_list = []
    done = False
    r_sum = 0
    while not done:

		# Escolhe a ação a ser tomada
		# Ou aleatória, ou a melhor ação como previsto pela rede
        eps *= decay_factor
        
        state = env.get_state()
        h_list.append(state[0])

        
        if np.random.random() < eps:
            choice = np.random.randint(0, 4)
            
        else:
            choice = np.argmax(model.predict(state.reshape(1,grid_size,grid_size,model_channels)))
            """ pred = model.predict(state.reshape(1,grid_size,grid_size,model_channels))[0]
            pred = pred - min(pred)
            pred = 4*pred/np.max(np.abs(pred))

            ex = np.exp(pred-np.max(pred))
            ex = ex/np.sum(ex)
            choice = np.random.choice([0,1,2,3],p=ex) """

        moves_table = [[1,0],[-1,0],[0,1],[0,-1]]
        [x,y] = moves_table[choice]
        move_type_table = ['right','left','down','up']
        move_type = move_type_table[choice]
        
        
        # Atua a ação no enviroment
        next_state, reward, done, _ = env.action(x,y)

        pred = model.predict(next_state.reshape(1,grid_size,grid_size,model_channels))

        target = reward + gamma * np.amax(pred[0])
        
        

		# Define o 'target', valor que queremos que a rede neural consiga prever
        target_vec = model.predict(state.reshape(1,grid_size,grid_size,model_channels))[0]
        #target_vec = np.clip(target_vec, -10, 10)
        target_vec[choice] = target

		# Adiciona a experiência na memória
        memory.add(state,target_vec)

		
        state = next_state
        r_sum += reward

        # Retira 50 amostras aleatórias da memória, e treina a rede nesse batch
        [states, targets] = memory.sample(100)
        model.train_on_batch(states.reshape(-1,grid_size,grid_size,model_channels), targets)
        #print(time.time()-lt)
        #lt = time.time()
        if render == True:
            env.render()

    env.reset()
    print('Iteration: ',i, '/   Reward: ', round(r_sum,3), '/  Eps = ',round(eps,5))
    r_list.append(r_sum)

    if i % 1000 == 0:
	    # Salva o modelo treinado para ser utilizado depois
        model.save_weights('model_weights.h5')

# Salva o modelo treinado para ser utilizado depois
model.save_weights('model_weights.h5')

# Plot de reward vs simulações
plt.figure(0)
plt.plot(r_list)


plt.show()


print('Finished')