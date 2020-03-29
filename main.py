

#import modules
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
from slmc import SLMCUpdater

# Name of parameter file (temporary)
param_file = "params/params.json"

# Parameters from json file
with open(param_file) as paramfile:
    params = json.load(paramfile)

# Assign variables from paramater file
#global variables (temporary)
J_factor = params["J factor"] #nearest-neighbor term factor
K_factor = params["K factor"] #plaquette term factor
size = params["size"] #size of state in 1D
chain_length = params["chain length"] #number of states to generate
T_start = params["T start"] #starting temperature
KLplot = params["KL plot"] #whether or not to calculate and plot KL distance
if size>4: #do not calculate KL distance if larger than 4x4
    KLplot = False


#calculate spin correlation
def spin_correlation(state):
    horiz_shift = np.insert(state[:,1:], len(state[0,:])-1, state[:,0], axis=1) #horizonatally shifted state
    vert_shift = np.insert(state[1:], len(state[:,0])-1, state[0], axis=0) #vertically shifted state
    
    return np.sum( state*horiz_shift + state*vert_shift )

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

#effective hamiltonian
def h_eff(state, E0, J1):
    e_neighbor = spin_correlation(state)
    return E0 - J1*e_neighbor

#KL distance between probability set 1 and 2
def KLdistance(prob1, prob2):
    return np.sum( prob1 * np.log(prob1 / prob2) )


#calculate thermodynamic beta
k_b = 8.617e-5 #Boltzmann constant in eV/K
beta = 1/(k_b * T_start)


#generate all possible states to compute KL distance
#don't run this if larger than 4x4
if KLplot:
    states=[]
    fmt_str = '{0:0'+str(size**2)+'b}' #converts int to binary
    for i in range(2**(size**2)): #loop over length of grid
        bin_str = list(fmt_str.format(i)) 
        bin_str = np.reshape(bin_str,(size,size)).astype('int')
        bin_str[bin_str==0] = -1
        states.append( bin_str )
    
    #calculate partition function
    e_lambdas = [np.exp(-beta*hamiltonian(s)) for s in states]
    z_lambda = np.sum(e_lambdas)
    p_lambdas = e_lambdas / z_lambda


        
#STEP 1: generate states with local update
local = LocalUpdater(hamiltonian, beta)
wolff = WolffUpdater(J_factor, beta)

initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
initial_state[initial_state==0] = -1 #replace 0s with -1s

#markov chain of generated states
state_chain = []
state_chain.append(initial_state)

E_alphas = [] #list of energies
C_alphas = [] #list of spin correlations
E_alphas.append(hamiltonian(initial_state))
C_alphas.append(spin_correlation(initial_state))

kls = []

start_time = time.time()
for n in range(chain_length):
    state = np.copy(state_chain[n])

    #perform local update
    state = local.update(state)
            
    #add state to chain
    state_chain.append(state)
    E_alphas.append(hamiltonian(state))
    C_alphas.append(spin_correlation(state))
    
    #calculate KL distance
    if KLplot:
        #probability of generated states
        gen_states, gen_freq = np.unique(state_chain, return_counts=True, axis=0) #unique states in chain
        prob_gen = gen_freq.astype('float') / np.sum(gen_freq)
        
        #model probabilities        
        e_lambdas = [np.exp(-beta*hamiltonian(s)) for s in gen_states]
        p_lambdas = e_lambdas / z_lambda
       
        #calculate KL distance
        kl = KLdistance(prob_gen, p_lambdas)
        kls.append(kl)

print(time.time()-start_time)

if KLplot:
    plt.plot(np.arange(len(kls)), kls, 'k')
    plt.xlabel('number of states', fontsize=16)
    plt.ylabel('KL distance', fontsize=16)
    plt.title('KL distance '+str(size)+'x'+str(size)+' grid', fontsize=16)
    #plt.savefig('KLdistance'+str(size)+'x'+str(size))
    plt.show()
    plt.clf()

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

#change spin-correlation and energy for plotting
idx = np.where(C_alphas >= 0)
C_plot = C_alphas[idx] / float(size**2)
E_plot = E_alphas[idx] / float(size**2)
E_eff_plot = E_effs[idx] / float(size**2)

#plot effective hamiltionian fit
plt.plot(C_plot, E_plot, 'ko')
plt.plot(C_plot, E_eff_plot, 'g')
plt.xlabel(r'$C_1/N$', fontsize=16)
plt.ylabel(r'$E/N$', fontsize=16)
#plt.savefig('H-eff fit', bbox_inches='tight')
plt.show()
plt.clf()


#STEP 3 and 4: Perform SLMC Wolff update
T_slmc = 3. #temperature to perform SLMC at
beta = 1/(k_b * T_slmc) #new beta
kwargs = {'E0':E0, 'J1':J1} #arguments for h_eff in slmc class

slmc = SLMCUpdater(J1, beta, hamiltonian, h_eff, **kwargs)

state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
state[state==0] = -1 #replace 0s with -1s

state = slmc.update(state)

