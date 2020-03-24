
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from mpmath import csch
import time

#global variables (temporary)
J_factor = 1. #nearest-neighbor term factor
K_factor = 0.2 #plaquette term factor
size = 10 #size of state in 1D
chain_length = 1000 #number of states to generate
k_b = 8.617e-5 #Boltzmann constant in eV/K

def mag_function(T):
    c = csch(2*J_factor / (k_b*T))
    return (1-c**2)**(1./8.)

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
    counter = 0
    mags = np.zeros( shape=(chain_length+1,1) ) #[], changed to np to use * in the mags_autocorr calculation later 
    for s in state_chain: 
        mags[counter] =  abs(np.sum(s)) / (size**2)  # of size 1001 at end 
        counter = counter + 1
        
    mags_autocorr = np.zeros( shape=(chain_length+1,1) )
    for dt in range(0,chain_length+1): 
        mags_autocorr[dt] = np.sum( mags[0:(chain_length+1-dt)]*mags[(0+dt):chain_length+1] ) / (chain_length+1-dt)
        
    state_chain=state_chain[700:]
    mags = mags[700:]
    

    return np.mean(mags), mags_autocorr # avg mags over later time steps, and autocorrelation is over all time steps 
    
    
    
Ts = np.arange(5000, 55000, 5000)


size = 10 #size of state in 1D
mags = []
for T in Ts:
    mean_mag, mags_autocorr = calc_mag(T)
    mags.append(mean_mag)
    plt.plot(mags_autocorr - mean_mag**2)
#p1,=plt.plot(Ts, mags, 'ko')
plt.xlabel(r'$dt$', fontsize=16)
plt.ylabel(r'$M(t)M(t+dt) - M^2$', fontsize=16)
plt.savefig('Fig3ofPaper_local')

'''
size = 5 #size of state in 1D
mags = []
for T in Ts:
    mags.append(calc_mag(T))
p2,=plt.plot(Ts, mags, 'go')

size = 8 #size of state in 1D
mags = []
for T in Ts:
    mags.append(calc_mag(T))
p3,=plt.plot(Ts, mags, 'bo')

plt.legend([p1,p2,p3],['3x3','5x5','8x8'])

size = 3 #size of state in 1D
start_time = time.time()
mean_mag, mags_autocorr = calc_mag(Ts[0]) 
mean_mag2 = mean_mag**2


mags = []
for T in Ts:
    mags.append(calc_mag(T))
p3,=plt.plot(Ts, mags, 'bo')

T_c = (2/np.log(1+np.sqrt(2))) * (J_factor/k_b)
print(T_c)

#plot magnetization function
step = (Ts[-1] - Ts[0])/100
plot_t = np.arange(Ts[0], Ts[-1]+step, step)
mag_plot=[]
for t in plot_t:
    if t<T_c:
        mag_plot.append(mag_function(t))
    else:
        mag_plot.append(0)
plt.plot(plot_t, mag_plot, 'm')

plt.plot([T_c, T_c], [0,1], 'r--')

plt.xlabel(r'$T$', fontsize=16)
plt.ylabel(r'$m$', fontsize=16)
plt.savefig('magnetization')

'''