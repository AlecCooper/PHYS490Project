
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater


#global variables (temporary)
J_factor = 1.0e-4 #nearest-neighbor term factor
K_factor = 0.2e-4 #plaquette term factor
size = 5 #size of state in 1D
chain_length = 2000 #number of states to generate
k_b = 8.617e-5 #Boltzmann constant in eV/K

#original hamiltonian
def hamiltonian(state, i_ind=None, j_ind=None):            
    if i_ind==None and j_ind==None: #if no indices specified
        #generate shifted matrices
        horiz_shift = np.insert(state[:,1:], len(state[0,:])-1, state[:,0], axis=1)
        vert_shift = np.insert(state[1:], len(state[:,0])-1, state[0], axis=0)
        diag_shift = np.insert(horiz_shift[1:], len(horiz_shift[:,0])-1, horiz_shift[0], axis=0)
        
        #nearest-neighbor term
        e_neighbor = np.sum( state*horiz_shift + state*vert_shift )
        
        #plaquette term
        e_plaquette = np.sum( state*horiz_shift*vert_shift*diag_shift )

    else: #for a single index
        state_x = len(state[:,0])
        state_y = len(state[0,:])
    
        #nearest-neighbor term
        e_neighbor = 0 #initialize
        for i in [i_ind-1, i_ind]:
            for j in [j_ind-1, j_ind]:
                e_neighbor += state[i,j] * state[i, (j+1) % state_y] #horizontal term
                e_neighbor += state[i,j] * state[(i+1) % state_x, j] #vertical term
        
        #plaquette term
        e_plaquette = 0 #initialize
        for i in [i_ind-1, i_ind]:
            for j in [j_ind-1, j_ind]:
                plaquette = state[i,j] * state[i, (j+1) % state_y]
                plaquette *= state[(i+1) % state_x, j] * state[(i+1) % state_x, (j+1) % state_y]
                e_plaquette += plaquette

    return -J_factor*e_neighbor - K_factor*e_plaquette

#calculate magnetization for a given temperature
def calc_mag(T):
    beta = 1/(k_b * T) #thermodynamic beta

    local = LocalUpdater(hamiltonian, beta)
    #local = WolffUpdater(J_factor, beta)

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
    state_chain=state_chain[1500:]


    mags=[]
    for s in state_chain:
        mags.append(abs(np.sum(s)) / float(size**2))
    
    print(size, T)
    
    return np.mean(mags)
    
    
Ts = np.arange(15000, 34000, 1000)

Ts = np.arange(0.5,9.,0.5)

size = 5 #size of state in 1D
mags = []
for T in Ts:
    mags.append(calc_mag(T))
p1,=plt.plot(Ts, mags, 'ko')
'''
size = 10 #size of state in 1D
mags = []
for T in Ts:
    mags.append(calc_mag(T))
p2,=plt.plot(Ts, mags, 'go')

size = 15 #size of state in 1D
mags = []
for T in Ts:
    mags.append(calc_mag(T))
p3,=plt.plot(Ts, mags, 'bo')

plt.legend([p1,p2,p3],['5x5','10x10','15x15'])
'''


T_c = (2/np.log(1+np.sqrt(2))) * (J_factor/k_b)
print(T_c)

plt.plot([T_c, T_c], [0,1], 'r--')


plt.xlabel(r'$T$', fontsize=16)
plt.ylabel(r'$m$', fontsize=16)
plt.savefig('magnetization')

