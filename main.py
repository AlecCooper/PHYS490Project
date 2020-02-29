

#import modules
import numpy as np


#global variables (temporary)
J_factor = 1. #nearest-neighbor term factor
K_factor = 2. #plaquette term factor
size = 3 #size of state in 1D
chain_length = 10 #number of states to generate
T_start = 1000. #starting temperature

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

#STEP 1: generate states with local update

#calculate thermodynamic beta
k_b = 8.617e-5 #Boltzmann constant in eV/K
beta = k_b * T_start

#markov chain of generated states
state_chain = []
state_chain.append(initial_state)

for n in range(chain_length):
    state = np.copy(state_chain[n])
    energy = hamiltonian(state)
    
    #loop over sites in state to perform local update    
    for i in range(size):
        for j in range(size):
            accept = False #whether or not flip is accepted
            state[i,j] *= (-1) #try flipping bit
            flipped_energy = hamiltonian(state) #energy of flipped state
            
            if flipped_energy < energy: #accept if lower energy
                accept = True
            else: #if higher energy
                p_rand = np.random.uniform(0, 1)
                prob_accept = np.exp(-1*beta*(flipped_energy-energy)) #probability of accepting
                if p_rand < prob_accept: #accept if lower than probability
                    accept = True
            
            if accept==True: #if flip is accepted
                energy = flipped_energy #update energy
            else: #if rejected
                state[i,j] *= (-1) #flip bit back
            
    #add state to chain
    state_chain.append(state)
    
for i in range(len(state_chain)):
    print(state_chain[i])



    