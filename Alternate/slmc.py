# class that performs self-learning monte carlo update on a given state
# returns the next state in the Markcv chain

#import modules
import numpy as np
import copy
from wolffupdate import WolffUpdater

class SLMCUpdater():
    
    def __init__(self, hamiltonian, h_eff, J_factor, K_factor, beta, J_1, E_0):
        self.beta = beta
        self.J_factor = J_factor
        self.K_factor = K_factor
        self.hamiltonian = hamiltonian
        self.h_eff = h_eff
        self.J_1 = J_1
        self.E_0 = E_0

    def update(self, old_state):

        state = copy.deepcopy(old_state)

        # Generate a cluster with the wolff updater
        wolff = copy.deepcopy(WolffUpdater(self.J_factor, self.beta))
        state = wolff.update(state)

        # Calculate energy and effective energy of old state
        e_a = self.hamiltonian(old_state,self.J_factor,self.K_factor)
        #print(e_a)
        e_a_effective = self.h_eff(old_state,self.E_0,self.J_1)
        #print(e_a_effective)

        # Calculate energy and effective energy of new state
        e_b = self.hamiltonian(state,self.J_factor,self.K_factor)
        e_b_effective = self.h_eff(state,self.E_0,self.J_1)

        # Probability of accepting the new state
        prob = min(1., np.exp(-1.*self.beta*(e_b-e_b_effective)-(e_a-e_a_effective)))

        # Accept of reject
        x = np.random.uniform()
        if x < prob:
            return state
        
        else:
            return old_state



        


    
    
    
