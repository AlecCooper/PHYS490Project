# class that performs local update on a given state
# returns the next state in the Markcv chain

#import modules
import numpy as np
import random as rand
import copy

class LocalUpdater():
    
    def __init__(self, hamiltonian, beta, J_factor, K_factor):
        self.hamiltonian = hamiltonian
        self.beta = beta
        self.J_factor = J_factor
        self.K_factor = K_factor

    def change_p(self, beta):
        self.beta = beta
    
    def update(self, old_state):
        
        state = copy.deepcopy(old_state)

        N = len(state)

        # Generate randomly ordered list of sites
        sites = []
        for i in range(N):
            for j in range(N):
                sites.append((i,j))
        sites = rand.sample(sites,N)

        # N is Markov chain flip attempts
        for n in range(N):

            # Pick random site
            site = sites[n]

            # Calculate energy
            energy = self.hamiltonian(state, self.J_factor, self.K_factor, site)
            #energy = self.hamiltonian(state, self.J_factor, self.K_factor)

            # Flip site
            flipped_state = copy.deepcopy(state)
            flipped_state[site] *= -1

            # Calculate energy of flipped site
            flipped_energy = self.hamiltonian(flipped_state, self.J_factor, self.K_factor, site)
            #flipped_energy = self.hamiltonian(flipped_state, self.J_factor, self.K_factor)

            if flipped_energy <= energy: #accept if lower energy
                state = flipped_state
            
            else:
                # Calculate random var x
                x = np.random.uniform()
                if np.exp(-1.*self.beta*(flipped_energy - energy)) > x: #prob of flipping site
                    state = flipped_state
                    

        return state