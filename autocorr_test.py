
#test plotting magnetization to see phase transition

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
from slmc import SLMCUpdater
from scipy.optimize import curve_fit
import time


#global variables (temporary)
J_factor = 1.0e-4 #nearest-neighbor term factor
K_factor = 0.2e-4 #plaquette term factor
size = 20 #size of state in 1D
chain_length = 50000 #number of states to generate
burn_in = 1000 #number of steps in burn in
max_dt = 1000 #maximum dt in autocorrelation plot
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

#calculate autocorrelation
def calc_autocorr(state_chain):
    #calculate list of magnetizations per site
    mags=[] #list of magnetizations
    for i in range(len(state_chain)):
        mags.append(abs(np.sum(state_chain[i])) / float(size**2))

    #plt.plot(range(len(mags)),mags,'k')
    #plt.show()
    #plt.clf()

    '''
    mags = np.array(mags[burn_in:]) #trim off burn-in
    auto_corr = np.zeros(max_dt)
    mu = np.mean(mags)
    for s in range(1, max_dt):
        auto_corr[s] = np.mean( (mags[:-s]-mu) * (mags[s:]-mu) ) #/ np.var(mags)
    return mu , auto_corr
    '''

    mags = np.array(mags[burn_in:]) #trim off burn-in
    mag = np.mean(mags) #magnetization
    mag_autocorr = []
    for dt in range(1, max_dt):
        mag_prod = mags[:-dt] * mags[dt:] #product of magnetizations
        mag_autocorr.append(np.mean(mag_prod) - mag**2) #autocorrelation for delta_t
    
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
    T_start = T
    beta = 1/(k_b * T_start) #thermodynamic beta

    #STEP 1: generate states with local update
    local = LocalUpdater(hamiltonian, beta)

    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
      
    #markov chain of generated states
    state_chain = []
    state_chain.append(initial_state)
    
    start_time = time.time()
    for n in range(chain_length):
        state = np.copy(state_chain[n])
    
        #perform local update
        state = local.update(state)
        
        #add state to chain
        state_chain.append(state)
        
        #print steps occaisonally
        if (n%1000)==0:
            print(n, '{0:.3f}.'.format(time.time()-start_time))


    #generate mag chains
    _, local_autocorr = calc_autocorr(state_chain)
    
    return local_autocorr



T_c = (2/np.log(1+np.sqrt(2))) * (J_factor/k_b)
print(T_c)

mode = 0 #0=calculate, 1=plot

if mode==0: #calculate and output autocorrelation    
    T = 2.5
    
    local_autocorr = calc_mag(T)
    
    output_array=np.array([range(len(local_autocorr)),local_autocorr])
    np.savetxt('autocorrelation'+str(T)+'.csv', output_array, delimiter=',')
    
else: #plot autocorrelation
    Ts = np.array([2.5])
    
    ps = []
    legend_labels = []
    colors=['k','g','b']
    for i in range(len(Ts)):
        T=Ts[i]
            
        #read in data
        in_data = np.genfromtxt('autocorrelation'+str(T)+'.csv', delimiter=',')
        local_autocorr = in_data[1]
        local_autocorr /= max(local_autocorr) #normalize
        
        p,=plt.plot(range(len(local_autocorr)), local_autocorr, colors[i])
        ps.append(p)
        legend_labels.append(str(T))#str(T/T_c)+r' $T_c$')
    
    #format plot and label
    plt.legend(ps,legend_labels)
    plt.xlabel(r'$dt$', fontsize=16)
    plt.ylabel(r'$\langle M(t)M(t+dt) \rangle - \langle M \rangle ^2$', fontsize=16)
    
    plt.savefig('autocorrelation_compare', bbox_inches='tight')


