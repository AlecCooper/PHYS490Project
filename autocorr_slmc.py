
# this looks at the autocorrelation function of the self-learning wolff method 

#import modules
import numpy as np
import matplotlib.pyplot as plt
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
from slmc import SLMCUpdater
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
size = 10 #size of state in 1D
chain_length = 2000 #number of states to generate
k_b = 8.617e-5 #Boltzmann constant in eV/K

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

#calculate magnetization for a given temperature
def calc_mag(T):
    beta = 1/(k_b * T) #thermodynamic beta 
    #method = LocalUpdater(hamiltonian, beta)
    #method = WolffUpdater(J_factor, beta)
    
    #-------------------- if the method is self learning wolff--------------------
    #STEP 1: generate states with local update
    local = LocalUpdater(hamiltonian, beta)
    #wolff = WolffUpdater(J_factor, beta)
    
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
    
    method = SLMCUpdater(J1, beta, hamiltonian, h_eff, **kwargs)
    #----------------------------------------------------------------------
    
    initial_state = np.random.randint(0, 2, (size,size)) #initialize state from 0 to 1
    initial_state[initial_state==0] = -1 #replace 0s with -1s
      
    #markov chain of generated states
    state_chain = []
    state_chain.append(initial_state)  
    for n in range(chain_length):
        state = np.copy(state_chain[n])
    
        #perform update using the SLMC method 
        state = method.update(state)
        
        #add state to chain
        state_chain.append(state)
    
    #calculate magnetization
    state_chain=state_chain[1500:]

    mags=[] #list of magnetizations
    for i in range(len(state_chain)):
        mags.append(abs(np.sum(state_chain[i])) / float(size**2))
    mag = abs(np.mean(mags)) #magnetization
    
    mag_product=[] #product of current and next magnetization
    mag_autocorr=[] #autocorrelation function
    for i in range(len(state_chain)-1):
        mag_product.append(mags[i]*mags[i+1])
        
        mag_product_exp = np.mean(mag_product)
        mag_exp = np.mean(mags[:i])
        mag_autocorr.append(mag_product_exp - mag_exp**2)
 
    return mag, mag_autocorr


# for fitting the autocorrelation function to an exponential function
def func(x, a, b, c):
     return a * np.exp(-b * x) +c
 

#                   end of functions 
#------------------------------------------------------------------------------
     
 
    
Tc = 2.5
moment, mags_autocorr = calc_mag(Tc)
plt.plot(range(len(mags_autocorr)), mags_autocorr, 'ko')
plt.xlabel(r'$dt$', fontsize=16)
plt.ylabel(r'$M(t)M(t+dt) - M^2$', fontsize=16)
plt.show()
plt.savefig('autocorr_slmc')

mags_autocorr[0] = 0 # this had nan as the first value so I just set it to zero  
ind_max = np.argmax(mags_autocorr)   #find index of maximum value in mags_autocorr 
ydata = mags_autocorr[ind_max:]
xdata = np.linspace(0, len(ydata)-1, len(ydata)) 
popt, pcov = curve_fit(func, xdata, ydata) # perform curve fitting using scipy's non-linear least squares 

# plot of the fitted function and original data 
p1, = plt.plot(xdata, func(xdata, popt[0],popt[1],popt[2])) 
p2, = plt.plot(xdata, ydata)
plt.legend([p1,p2],['fit','original'])
plt.savefig('autocorr_slmc_fit')

# this is the autocorrelation time 
tau = 1/popt[1] 
print(tau)














