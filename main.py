

#import modules
import numpy as np
import matplotlib.pyplot as plt


#global variables (temporary)
J_factor = 1. #nearest-neighbor term factor
K_factor = 0.2 #plaquette term factor
size = 5 #size of state in 1D
chain_length = 100 #number of states to generate
T_start = 1000. #starting temperature

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


initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
initial_state[initial_state==0] = -1 #replace 0s with -1s


#STEP 1: generate states with local update

#calculate thermodynamic beta
k_b = 8.617e-5 #Boltzmann constant in eV/K
beta = k_b * T_start

#markov chain of generated states
state_chain = []
state_chain.append(initial_state)

E_alphas = [] #list of energies
C_alphas = [] #list of spin correlations
E_alphas.append(hamiltonian(initial_state))
C_alphas.append(spin_correlation(initial_state))

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
    E_alphas.append(hamiltonian(state))
    C_alphas.append(spin_correlation(state))
    
#convert to np arrays
E_alphas = np.reshape(np.array(E_alphas) , (len(E_alphas),1))
C_alphas = np.reshape(np.array(C_alphas) , (len(C_alphas),1))


#STEP 2: learn effective hamiltionian

#perform linear regression to learn effective hamiltonian
Phi = np.insert(C_alphas,0,np.ones(len(C_alphas.T[0])),axis=1) #pad with ones
MPinverse = np.linalg.inv( np.matmul(Phi.T, Phi) )
MPinverse = np.matmul(MPinverse, Phi.T)
w_linear = np.matmul(MPinverse, E_alphas)

E0 = w_linear[0][0]
J1 = -w_linear[1][0]
print(E0, J1)

#calculate energy with H_eff
E_effs = []
for s in state_chain:
    E_effs.append(h_eff(s, E0, J1))
E_effs = np.reshape(np.array(E_effs) , (len(E_effs),1))

#calculate mean error
mean_err = np.mean(abs(E_alphas - E_effs))
print(mean_err)

#plot effective hamiltionian fit
plt.plot(C_alphas, E_alphas, 'ko')
plt.plot(C_alphas, E_effs, 'g')
plt.xlabel(r'$C_n^{\alpha}$', fontsize=16)
plt.ylabel(r'$E^{\alpha}$', fontsize=16)
plt.show()
