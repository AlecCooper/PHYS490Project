
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
from slmc import SLMCUpdater
from scipy.optimize import curve_fit


#global variables (temporary)
J_factor = 1.0e-4 #nearest-neighbor term factor
K_factor = 0.2e-4 #plaquette term factor
size = 15 #size of state in 1D
chain_length = 1000 #number of states to generate
k_b = 8.617e-5 #Boltzmann constant in eV/K


#function that describes autocorrelation time
def autocorrelation_function(x, a, b, c, t):
    return a*np.exp(-t * (x - b)) + c

#plot autocorrelation and return autocorrelation time
def plot_autocorrelation(autocorr_chain, color):
    #start fit at peak
    start_id = autocorr_chain.index(np.nanmax(autocorr_chain))

    #perform fit
    plot_xs = np.arange(len(autocorr_chain[start_id:]))+start_id
    par, cov = curve_fit(autocorrelation_function, plot_xs, autocorr_chain[start_id:], p0=[0.1,start_id,0.02,1./50.])
    print(par)
    
    #calculate plot data points
    plot_ys=[]
    for x in plot_xs:
        y = autocorrelation_function(x, par[0], par[1], par[2], par[3])
        plot_ys.append(y)
        
    #plot fit
    plt.plot(plot_xs, plot_ys, color)
    
    #return autocorrelation time
    auto_time = 1./par[3]
    return auto_time
    
#calculate spin correlation
def spin_correlation(state):
    horiz_shift = np.insert(state[:,1:], len(state[0,:])-1, state[:,0], axis=1) #horizonatally shifted state
    vert_shift = np.insert(state[1:], len(state[:,0])-1, state[0], axis=0) #vertically shifted state
    
    return np.sum( state*horiz_shift + state*vert_shift )

#effective hamiltonian
def h_eff(state, E0, J1):
    e_neighbor = spin_correlation(state)
    return E0 - J1*e_neighbor

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

#calculate autocorrelation
def calc_autocorr(state_chain):
    #calculate list of magnetizations per site
    mags=[] #list of magnetizations
    for i in range(len(state_chain)):
        mags.append(abs(np.sum(state_chain[i])) / float(size**2))

    start_t = 100 #initial time
    chain_length = len(state_chain) - start_t
    mags = np.array(mags[start_t:])
    mag = np.mean(mags) #magnetization
    
    delta_ts = range(1, chain_length - start_t) #range of delta ts
    mag_autocorr = []
    for dt in delta_ts:
        mag_prod = mags[ : chain_length-dt] * mags[dt : chain_length] #product of magnetizations
        mag_autocorr.append(np.mean(mag_prod))# - mag**2) #autocorrelation for delta_t
    
    return mag, mag_autocorr
    
    '''
    mag_product=[] #product of current and next magnetization
    mag_autocorr=[] #autocorrelation function
    for i in range(len(state_chain)-1):
        mag_product.append(mags[i]*mags[i+1])
        
        mag_product_exp = np.mean(mag_product)
        mag_exp = np.mean(mags[:i])
        mag_autocorr.append(mag_product_exp - mag_exp**2)
 
    return mag, mag_autocorr
    '''

#calculate magnetization for a given temperature
def calc_mag(T):
    T_start = T + 3.
    beta = 1/(k_b * T_start) #thermodynamic beta

    #STEP 1: generate states with local update
    local = LocalUpdater(hamiltonian, beta)

    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
      
    #markov chain of generated states
    state_chain = []
    state_chain.append(initial_state)
    E_alphas = [] #list of energies
    C_alphas = [] #list of spin correlations
    for n in range(chain_length):
        state = np.copy(state_chain[n])
    
        #perform local update
        state = local.update(state)
        
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
    kwargs = {'E0':E0, 'J1':J1} #arguments for h_eff in slmc class
    
    #STEP 3 and 4: perform SLMC
    beta = 1/(k_b * T) #new beta
    
    #initialize updaters
    local = LocalUpdater(hamiltonian, beta)
    wolff = WolffUpdater(J_factor, beta)
    slmc = SLMCUpdater(J1, beta, hamiltonian, h_eff, **kwargs)
    
    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
    
    local = LocalUpdater(hamiltonian, beta)
    wolff = WolffUpdater(J_factor, beta)
    
    #markov chain of generated states
    local_chain = []
    local_chain.append(initial_state)
    wolff_chain = []
    wolff_chain.append(initial_state)
    slmc_chain = []
    slmc_chain.append(initial_state)
    
    #generate state chains for each update method
    for n in range(chain_length):
        local_state = np.copy(local_chain[n])
        wolff_state = np.copy(wolff_chain[n])
        slmc_state = np.copy(slmc_chain[n])
    
        #perform local update
        local_state = local.update(local_state)
        wolff_state = wolff.update(wolff_state)
        slmc_state = slmc.update(slmc_state)        
        
        #add state to chain
        local_chain.append(local_state)
        wolff_chain.append(wolff_state)
        slmc_chain.append(slmc_state)
        
    #generate mag chains
    _, local_autocorr = calc_autocorr(local_chain)   
    _, wolff_autocorr = calc_autocorr(wolff_chain)   
    _, slmc_autocorr = calc_autocorr(slmc_chain)    
    
    return local_autocorr, wolff_autocorr, slmc_autocorr



T_c = (2/np.log(1+np.sqrt(2))) * (J_factor/k_b)
print(T_c)

T = 2.6

local_autocorr, wolff_autocorr, slmc_autocorr = calc_mag(T)
colors = ['k', 'g', 'b']

p1,=plt.plot(range(len(local_autocorr)), local_autocorr, colors[0])
p2,=plt.plot(range(len(wolff_autocorr)), wolff_autocorr, colors[1])
p3,=plt.plot(range(len(slmc_autocorr)), slmc_autocorr, colors[2])

'''
#perform autocorrelation fits
local_time = plot_autocorrelation(local_autocorr, colors[0])
wolff_time = plot_autocorrelation(wolff_autocorr, colors[1])
slmc_time = plot_autocorrelation(slmc_autocorr, colors[2])
print(local_time, wolff_time, slmc_time)
'''

#format plot and label
plt.legend([p1,p2,p3], ['local','wolff','slmc'])
plt.xlabel(r'$dt$', fontsize=16)
plt.ylabel(r'$\langle M(t)M(t+dt) \rangle - \langle M \rangle ^2$', fontsize=16)

plt.savefig('autocorrelation_compare', bbox_inches='tight')




