# class that performs Wolff-cluster update on a given state
# returns the next state in the Markcv chain

#import modules
import numpy as np

class SLMCUpdater():
    
    state = []
    cluster_sites = []
    
    def __init__(self, J_factor, beta, hamiltonian, h_eff, **kwargs):
        self.e_const = 2.*J_factor*beta #factor for link probability
        self.beta = beta

        self.hamiltonian = hamiltonian #original hamiltonian
        self.h_eff = h_eff #effective hamiltonian
        self.kwargs = kwargs #arguments for h_eff
    
    #calculate probability of link
    def calc_prob(self, site1, site2):
        return np.max([0, 1-np.exp(self.e_const*site1*site2)])
    
    #add site to cluster
    def add_site(self, i1, j1, i2, j2):
        size = len(self.state)
        
        prob = self.calc_prob(self.state[i1,j1], self.state[i2,j2]) #probability of accepted
        p_rand = np.random.uniform(0, 1) #random number
        
        #if link is accepted and if site not already in cluster
        if p_rand < prob and (i2,j2) not in self.cluster_sites:
            self.state[i2,j2]*= (-1) #flip state
            self.cluster_sites.append((i2,j2)) #add new site to cluster
            
            #check adjacent states
            self.add_site(i2,j2,i2,(j2+1) % size)
            self.add_site(i2,j2,i2,(j2-1) % size)
            self.add_site(i2,j2,(i2+1) % size,j2)
            self.add_site(i2,j2,(i2-1) % size,j2)
    
    #update given state
    def update(self, state):
        self.state = np.copy(state) #update state
        size = len(state)
        
        #randomly select a site
        i,j = np.random.randint(0, size, size=2)
        self.state[i,j] *= (-1) #flip site
        self.cluster_sites=[(i,j)]

        #check adjacent states
        self.add_site(i,j,i,(j+1) % size)
        self.add_site(i,j,i,(j-1) % size)
        self.add_site(i,j,(i+1) % size,j)
        self.add_site(i,j,(i-1) % size,j)
            
        #decide whether or not to accept the update    
        e_diffB = self.hamiltonian(self.state) - self.h_eff(self.state, **self.kwargs)
        e_diffA = self.hamiltonian(state) - self.h_eff(state, **self.kwargs)
        p_exp = np.exp(-self.beta * (e_diffB - e_diffA))
        prob = min(1, p_exp)
        p_rand = np.random.uniform(0, 1) #random number
        if p_rand < prob:
            return_state = self.state #return new state
        else:
            return_state = state #return old state
        
        return return_state
    
