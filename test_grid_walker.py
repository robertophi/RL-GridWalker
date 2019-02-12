import grid_walker
import numpy as np
import time

env = grid_walker.GridWalker(2)
env.setup_gui()
env.render()

for i in range(0,1000):
    move = input("\nType input:")
    if move == 'a':
        [x,y] = [0,1]
    if move == 'b':
        [x,y] = [0,-1]
    if move == 'c':
        [x,y] = [1,0]
    if move == 'd':
        [x,y] = [-1,0]
    [state,reward, done, a] =env.action(x,y)
    #print(reward)
    #time.sleep(0.)
    env.render()


print('Finished')