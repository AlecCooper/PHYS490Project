
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt


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
            e_neighbor += J_factor * state[i,j] * state[i, (j+1) % state_y] #horizontal term
            e_neighbor += J_factor * state[i,j] * state[(i+1) % state_x, j] #vertical term

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

#effective hamiltonian
def h_eff(state, E0, J1):
    e_neighbor = spin_correlation(state)
    return E0 - J1*e_neighbor

#calculate magnetization for a given temperature
def calc_mag(T):
    beta = k_b * T #thermodynamic beta

    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
      
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
plt.show()

