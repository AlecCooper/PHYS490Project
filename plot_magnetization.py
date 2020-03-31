
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
from scipy.optimize import curve_fit

#function to fit to magnetization
def mag_function(x,a,b,c,e):
    less_func = lambda x: a*np.power(-x + b, 1./e) + c
    more_func = lambda x: -a*np.power((x - b), 1./e) + c
    y = np.piecewise(x, [x==b, x<b, x>b], [c, less_func, more_func])
    return y

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
    
    return np.mean(mags)
    

mode=0 #0=calculate, 1=plot

#temperature range
Ts = np.arange(0.5,9.,0.5)
step = (Ts[-1] - Ts[0]) / 100
T_plot = np.arange(Ts[0], Ts[-1]+step, step)

#linear size
size = 15

if mode==0: #calculate and output magnetization
    mags=[]
    for T in Ts:
        mags.append(calc_mag(T))
        print(size, T)
        
    output_array=np.array([Ts, mags])
    np.savetxt('magnetization_'+str(size)+'.csv', output_array, delimiter=',')
        
else: #plot magnetization
    sizes = [5,10,15]
    colors = ['k','g','b']

    ps=[]
    labels=[]
    for i in range(len(sizes)):
        size = sizes[i]
        
        #read in data
        in_data = np.genfromtxt('magnetization_'+str(size)+'.csv', delimiter=',')
        Ts = in_data[0]
        mags = in_data[1]
        
        p1,=plt.plot(Ts, mags, colors[i]+'o')
        ps.append(p1)
        labels.append(str(size)+'x'+str(size))
    
        #perform fit
        par, cov = curve_fit(mag_function, Ts[1:], mags[1:], p0=[1., 2.5, 0.5, 4.])
        print(par)
        
        ys = mag_function(T_plot, par[0],par[1],par[2],par[3])
        plt.plot(T_plot, ys, colors[i])
    
    #plot legend
    if len(sizes) > 1:
        plt.legend(ps, labels)

    #plot critical temperature
    #T_c = 2.6
    #plt.plot([T_c, T_c], [0,1], 'r--')    
    
    plt.xlabel(r'$T$', fontsize=16)
    plt.ylabel(r'$m$', fontsize=16)
    plt.savefig('autocorrelation_compare', bbox_inches='tight')

