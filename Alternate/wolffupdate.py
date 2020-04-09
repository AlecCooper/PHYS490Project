# class that performs Wolff-cluster update on a given state
# returns the next state in the Markcv chain

#import modules
import numpy as np
import copy

class WolffUpdater():
    
    def __init__(self, J_factor, beta):
         self.p = 1. - np.exp(-2.*J_factor*beta) #factor for link probability

    # update to a new value of p
    def change_p(self,J_factor,beta):
        self.p = 1. - np.exp(-2.*J_factor*beta)

    def update(self, old_state):

        state = copy.deepcopy(old_state)

        N = len(state)

        # Choose a random site
        site = (np.random.randint(0,N),np.random.randint(0,N))

        # Add site to cluster
        cluster = []
        cluster.append(site)
        new_sites = []
        new_sites.append(site)

        while len(new_sites) > 0:

            site = new_sites.pop()

            nearest_neighbors = []
            nearest_neighbors.append((site[0], (site[1] + 1) % N))
            nearest_neighbors.append((site[0], (site[1] - 1) % N))
            nearest_neighbors.append(((site[0] + 1) % N, site[1]))
            nearest_neighbors.append(((site[0] - 1) % N, site[1]))

            for neighbor in nearest_neighbors:
                if state[neighbor] == state[site]:
                    if not (neighbor in cluster):
                        x = np.random.uniform()
                        if x < self.p:
                            new_sites.append(neighbor)
                            cluster.append(neighbor)
                            #state[site] *= -1

        for site in cluster:
            state[site] *= -1

        return state



                



    
