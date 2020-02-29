

#import modules
import numpy as np


#global variables (temporary)
J_factor = 1.
K_factor = 2.
size = 3


#original hamiltonian
def hamiltonian(state):
    state_x = len(state[:,0])
    state_y = len(state[0,:])
    
    #nearest-neighbor term
    e_neighbor = 0 #initialize
    for i in range(state_x):
        for j in range(state_y):
            e_neighbor += J_factor * state[i,j] * state[i, (j+1) % state_y] #horizontal term
            e_neighbor += J_factor * state[i,j] * state[(i+1) % state_x, j] #vertical term
                    
    #plaquette term
    e_plaquette = 0 #initialize
    for i in range(state_x):
        for j in range(state_y):
            plaquette = state[i,j] * state[i, (j+1) % state_y]
            plaquette *= state[(i+1) % state_x, j] * state[(i+1) % state_x, (j+1) % state_y]
            e_plaquette += plaquette
    
    return -J_factor*e_neighbor - K_factor*e_plaquette



initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
initial_state[initial_state==0] = -1 #replace 0s with -1s

print(initial_state)
print hamiltonian(initial_state)


