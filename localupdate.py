# class that performs local update on a given state
# returns the next state in the Markcv chain

#faster version

#import modules
import numpy as np

class LocalUpdater():
    
    def __init__(self, hamiltonian, beta):
        self.hamiltonian = hamiltonian
        self.beta = beta
    
    def update(self, state):
        size = len(state)
        
        #generate random probability
        p_rand = np.random.uniform(0, 1, size=(size,size))
        
        #loop over sites in state to perform local update    
        for i in range(size):
            for j in range(size):
                accept = False #whether or not flip is accepted
                energy = self.hamiltonian(state, i, j)
                state[i,j] *= (-1) #try flipping bit
                flipped_energy = self.hamiltonian(state, i, j) #energy of flipped state
                
                if flipped_energy < energy: #accept if lower energy
                    accept = True
                else: #if higher energy
                    prob_accept = np.exp(-1*self.beta*(flipped_energy-energy)) #probability of accepting
                    if p_rand[i,j] < prob_accept: #accept if lower than probability
                        accept = True
                
                if accept==True: #if flip is rejected
                    energy = flipped_energy #update energy
                else: #if rejected
                    state[i,j] *= (-1) #flip bit back
                    
        return state #return updated state
