
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater


#global variables (temporary)
J_factor = 1. #nearest-neighbor term factor
K_factor = 0.2 #plaquette term factor
size = 3 #size of state in 1D
chain_length = 1000 #number of states to generate
k_b = 8.617e-5 #Boltzmann constant in eV/K

#calculate spin correlation
def spin_correlation(state):
    state_x = len(state[:,0])
    state_y = len(state[0,:])
    
    #nearest-neighbor term
    e_neighbor = 0 #initialize
    for i in range(state_x):
        for j in range(state_y):
            e_neighbor += state[i,j] * state[i, (j+1) % state_y] #horizontal term
            e_neighbor += state[i,j] * state[(i+1) % state_x, j] #vertical term

    return e_neighbor

#original hamiltonian
def hamiltonian(state):
    #nearest-neighbor term
    e_neighbor = spin_correlation(state)
    
    #plaquette term
    state_x = len(state[:,0])
    state_y = len(state[0,:])
    e_plaquette = 0 #initialize
    for i in range(state_x):
        for j in range(state_y):
            plaquette = state[i,j] * state[i, (j+1) % state_y]
            plaquette *= state[(i+1) % state_x, j] * state[(i+1) % state_x, (j+1) % state_y]
            e_plaquette += plaquette
    
    return -J_factor*e_neighbor - K_factor*e_plaquette

#calculate magnetization for a given temperature
def calc_mag(T):
    beta = k_b * T #thermodynamic beta

    local = LocalUpdater(hamiltonian, beta)

    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
      
    #markov chain of generated states
    state_chain = []
    state_chain.append(initial_state)  
    for n in range(chain_length):
        state = np.copy(state_chain[n])
    
        #perform local update
        state = local.update(state)
                
        #add state to chain
        state_chain.append(state)
        
    #calculate magnetization
    state_chain[700:]
    mags=[]
    for s in state_chain:
        mags.append(np.sum(s))
    
    return np.mean(mags) / (size*size)

betas = np.arange(0., 2., 0.1)
Ts = betas/k_b

mags = []
for T in Ts:
    mags.append(calc_mag(T))

plt.plot(Ts, mags, 'ko')


T_c = J_factor/(2.*k_b)
print(T_c)


plt.xlabel(r'$T$', fontsize=16)
plt.ylabel(r'$M$', fontsize=16)
plt.show()

