#import modules
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from localupdate import LocalUpdater
from wolffupdate import WolffUpdater
import sklearn.linear_model as lm
from slmc import SLMCUpdater

#original hamiltonian
def hamiltonian(state, J_factor, K_factor, site=None):

    if site==None:

        # compute shifted matricies
        hor_shift = np.roll(state,1,axis=1)
        vert_shift = np.roll(state,1,axis=0)
        diag_shift = np.roll(vert_shift,1,axis=1)

        # Placket term
        e_plaquette = np.multiply(np.multiply(state,hor_shift),np.multiply(vert_shift,diag_shift))
        e_plaquette = np.sum(e_plaquette)

        # Nearest neighbor term
        e_neighbor = np.multiply(hor_shift,state) + np.multiply(vert_shift,state)
        e_neighbor = np.sum(e_neighbor)
    
    else:
        # Nearest neighbor term
        N = len(state)
        e_neighbor = 0
        e_neighbor += (state[site]) * (state[site[0],(site[1]+1)%N])
        e_neighbor += (state[site]) * (state[site[0],(site[1]-1)%N])
        e_neighbor += (state[site]) * (state[(site[0]+1)%N,site[1]])
        e_neighbor += (state[site]) * (state[(site[0]-1)%N,site[1]])

        # Plaquette term
        def plaquette(state, site):
            N = len(state)
            e_plaquette = 1.
            e_plaquette *= state[site]
            e_plaquette *= state[site[0],(site[1]+1)%N]
            e_plaquette *= state[(site[0]+1)%N,(site[1]+1)%N]
            e_plaquette *= state[(site[0]+1)%N,site[1]]
            return e_plaquette

        e_plaquette = 0

        for i in (0,1):
            for j in (0,1):
                e_plaquette += plaquette(state,((site[0] + i)%N, (site[1] + j)%N))

    return -J_factor*e_neighbor - K_factor*e_plaquette

#calculate magnetization for a given temperature and state
def magnetization(state):
    mag = np.sum(state)
    return abs(mag)/(float(len(state)**2))

def plot_magnetization(initial_state, updater_type, grids=[5,10]):

    # Parameters from json file
    with open(param_file) as paramfile:
        params = json.load(paramfile)

    burn_in = params["burn in"]

    # list of magnetizations
    mags = []

    #generate range of temperatures
    temps = np.linspace(params["T start"], params["T end"],10)

    #J factor
    J_factor = params["J factor"]

    # init temp
    k_b = 8.617e-5 #Boltzmann constant in eV/K
    beta = 1./(k_b * temps[0])

    # Select updater
    if updater_type == "local":
        updater = LocalUpdater(hamiltonian, beta, params["J factor"], params["K factor"])
    elif updater_type == "wolff":
        updater = WolffUpdater(J_factor, beta)
    elif updater_type == "slmc":
        j_1, e_1 = train(temps[0], grids[0])
        updater = SLMCUpdater(hamiltonian, h_eff, J_factor, K_factor, beta, j_1, e_1)
    
    # loop through grids
    grids_mags = []
    for grid_size in grids:
        initial_state = np.random.choice([1,-1],size=(grid_size,grid_size))
        for temp in temps:
            # update temp
            k_b = 8.617e-5 #Boltzmann constant in eV/K
            beta = 1./(k_b * temp)

            # Change value of p
            if (updater_type == "wolff"):
                updater.change_p(J_factor, beta)
            elif (updater_type == "local"):
                updater.change_p(beta)
            elif (updater_type == "slmc"):
                j_1,e_0 = train(temp,grid_size)
                updater.J_1 = j_1
                updater.E_0 = e_0
                updater.beta = beta

            # calculate markov chain of states
            states = chain(initial_state, updater, params["chain length"])
            state_mags = []

            # Calculate magnetization for last generated states
            for state in states[burn_in:]:
                state_mags.append(magnetization(state))

            print(temp)
            print(grid_size) 

            mags.append(np.mean(state_mags))

        grids_mags.append(mags)
        mags = []

    plt.clf()
    ctr = 0
    for mags in grids_mags:
        label = "{} by {} grid".format(grids[ctr],grids[ctr])
        plt.plot(temps,mags, label=label)

        ctr += 1
    plt.legend()
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.show()

#calculate spin correlation
def spin_correlation(state):
    horiz_shift = np.insert(state[:,1:], len(state[0,:])-1, state[:,0], axis=1) #horizonatally shifted state
    vert_shift = np.insert(state[1:], len(state[:,0])-1, state[0], axis=0) #vertically shifted state
    
    return np.sum( state*horiz_shift + state*vert_shift )

#effective hamiltonian
def h_eff(state, E0, J1):
    e_neighbor = spin_correlation(state)
    return E0 - np.multiply(J1,e_neighbor)

# Generate a markov chain of states
def chain(initial_state, updater, chain_length):

    #markov chain of generated states
    state_chain = []
    state_chain.append(initial_state)

    for state_num in range(1,chain_length):

        state_chain.append(updater.update(state_chain[state_num-1]))

    return state_chain

# Trains the slmc updater
def train(temp,grid_size):

    # Name of parameter file (temporary)
    param_file = "params/params.json"

    # Parameters from json file
    with open(param_file) as paramfile:
        params = json.load(paramfile)

    # Assign variables from paramater file
    J_factor = params["J factor"] #nearest-neighbor term factor
    K_factor = params["K factor"] #plaquette term factor
    k_b = 8.617e-5 #Boltzmann constant in eV/K
    beta = 1/(k_b * temp)

    # Create a local updater to generate training data
    local = LocalUpdater(hamiltonian, beta, J_factor, K_factor)

    # Create a chain of states to use as training data
    initial_state = np.random.choice([1,-1],size=(grid_size,grid_size))
    state_chain = chain(initial_state, local, params["chain length"])

    # calculate spin correlations and energies
    corrs = []
    energies = []
    for state in state_chain:
        corrs.append(spin_correlation(state))
        energies.append(hamiltonian(state,J_factor,K_factor))

    corrs = np.array(corrs).reshape(-1,1)
    energies = np.array(energies).reshape(-1,1)

    # Preform linear regression
    lg = lm.LinearRegression().fit(corrs,energies)
    j_1 = lg.coef_[0,0]
    E_0 = lg.intercept_[0]

    return j_1, E_0

def train_intermediate(temp_sim,temp_crit,grid_size):

    # Name of parameter file (temporary)
    param_file = "params/params.json"

    # Parameters from json file
    with open(param_file) as paramfile:
        params = json.load(paramfile)

    # train model on training temp
    j_1, e_1 = train(temp_sim, grid_size)

    # Assign variables from paramater file
    J_factor = params["J factor"] #nearest-neighbor term factor
    K_factor = params["K factor"] #plaquette term factor
    k_b = 8.617e-5 #Boltzmann constant in eV/K
    beta = 1/(k_b * temp_crit)

    # Create our slmc
    slmc = SLMCUpdater(hamiltonian, h_eff, J_factor, K_factor, beta, j_1, e_1)

    # Create a chain of states to use as training data
    initial_state = np.random.choice([1,-1],size=(grid_size,grid_size))
    state_chain = chain(initial_state, slmc, params["chain length"])

    # calculate spin correlations and energies
    corrs = []
    energies = []
    for state in state_chain:
        corrs.append(spin_correlation(state))
        energies.append(hamiltonian(state,J_factor,K_factor))

    corrs = np.array(corrs).reshape(-1,1)
    energies = np.array(energies).reshape(-1,1)

    # Preform linear regression
    lg = lm.LinearRegression().fit(corrs,energies)
    j_1_new = lg.coef_[0,0]
    E_0_new = lg.intercept_[0]

    return j_1_new, E_0_new

def calc_autocorr(state_chain):
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
        
        if i%10==0:
            print(i)

    return mag_autocorr

if __name__ == "__main__":

    # Name of parameter file (temporary)
    param_file = "params/params.json"

    # Parameters from json file
    with open(param_file) as paramfile:
        params = json.load(paramfile)

    # Assign variables from paramater file
    J_factor = params["J factor"] #nearest-neighbor term factor
    K_factor = params["K factor"] #plaquette term factor
    size = params["size"] #size of state in 1D
    chain_length = params["chain length"] #number of states to generate
    T_start = params["T start"] #starting temperature
    T_end = params["T end"] #ending temperature
    KLplot = params["KL plot"] #whether or not to calculate and plot KL distance
    if size>4: #do not calculate KL distance if larger than 4x4
       KLplot = False


    initial_state = np.random.choice([1,-1],size=(size,size))

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
        e_lambdas = [np.exp(-beta*hamiltonian(s, J_factor, K_factor)) for s in states]
        z_lambda = np.sum(e_lambdas)
        p_lambdas = e_lambdas / z_lambda

            
    #STEP 1: generate states with local update
    local = LocalUpdater(hamiltonian, beta, J_factor, K_factor)
    wolff = WolffUpdater(J_factor, beta)
    j_1, e_1 = train_intermediate(5.,params["T start"],size)
    slmc = SLMCUpdater(hamiltonian, h_eff, J_factor, K_factor, beta, j_1, e_1)

    burn = params["burn in"]

    state_chain = chain(initial_state,slmc,chain_length)
    a_1 = calc_autocorr(state_chain)
    plt.plot(list(range(len(a_1))),a_1,label="slmc")
    state_chain = chain(initial_state,wolff,chain_length)
    a_2 = calc_autocorr(state_chain)
    plt.plot(list(range(len(a_2))),a_2,label="wolff")
    state_chain = chain(initial_state,local,chain_length)
    a_3 = calc_autocorr(state_chain)
    #plt.plot(list(range(len(a_3))),a_3,label="local")
    

    plt.legend()
    plt.show()

    #plot_magnetization(initial_state, "slmc")

   
    





    




