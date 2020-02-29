# -*- coding: utf-8 -*-

#main script for Phys 490 Assignment 3

#import relevant modules
import json, argparse
import numpy as np
import matplotlib.pyplot as plt

#convert spins from +/- to +1/-1
def convert_spin(state):
    #extract characters from string
    chars = list(state)

    #extract numerical spins
    spins=[]
    for c in chars:
        if c=='+':
            spins.append(1)
        else:
            spins.append(-1)
            
    return spins

#calculate the energy of a given state
def calc_energy(Js, spins):
    #sum couplers to calculate energy
    energy=0 
    for i in range(len(spins)):
        energy += Js[i] * spins[i] * spins[(i+1) % len(spins)]
        
    return -energy

#KL distance between probability set 1 and 2
def KLdistance(prob1, prob2):
    return np.sum( prob1 * np.log(prob1 / prob2) )


#assume beta
beta = 1.

#command line arguments
parser = argparse.ArgumentParser(description='Homework 3',
                                 
                                 epilog='hyper parameters from file \n' +                       
                                        '(parameter) \t (type) \t (default) \t (description) \n' +
                                        'learning rate \t float \t\t 0.001 \t\t learning rate of optimizer \n' +
                                        'batch size \t float \t\t 100 \t\t size of training batch \n' +
                                        'num epoch \t int \t\t 10000 \t\t number of training epochs \n' +
                                        'display epoch \t int \t\t 100 \t\t how often to display \n' +
                                        'test size \t int \t\t 100 \t\t size of test set' ,
                                        
                                 formatter_class=argparse.RawTextHelpFormatter)

#parser arguments
parser.add_argument('in_path', help='file path to input data')
parser.add_argument('--param', metavar='param.json',
                    help='parameter file name')
parser.add_argument('-v',
                       '--verbose',
                       action='store_true',
                       help='verbosity (optional)')
args = parser.parse_args()

#read in hyper parameters
if args.param is not None:
    paramfile = open(args.param)
    param = json.load(paramfile)
    learning_rate = param['learning rate']
    batch_size = param['batch size']
    num_epoch = param['num epoch']
    display_epoch = param['display epoch']
    test_size = param['test size']
else: #if no param file, use defaults
    learning_rate = 0.001
    batch_size = 100
    num_epoch = 10000
    display_epoch = 100
    test_size = 100

#read in data
data = np.loadtxt(args.in_path, dtype=str)

#separate train and test data sets
train_data = data[test_size:]
test_data = data[:test_size]
len_state = len(train_data[0]) #length of state

#generate all possible states
states=[]
fmt_str = '{0:0'+str(len_state)+'b}'
for i in range(2**len_state):
    bin_str = str(fmt_str.format(i))
    bin_str = bin_str.replace('1','+')
    bin_str = bin_str.replace('0','-')
    states.append(bin_str)

spin_states = [convert_spin(s) for s in states] #spin versions of states

#find frequency of states in test set
test_frequency = np.array([list(test_data).count(s) for s in states])
prob_test = test_frequency / np.sum(test_frequency)
prob_test[prob_test==0]=1e-8 #account for zero probabilities

#prepare training data and energies
training_spins = np.array([convert_spin(d) for d in train_data])
model_spins = np.copy(training_spins)
lambdas = np.random.uniform(-1, 1, len_state) *2 -1 #initialize lambdas from -1 to +1
energies = [calc_energy(lambdas, s) for s in training_spins]

#loop for each training epoch
kls=[] #stores KL distances
for epoch in range(1, num_epoch + 1):
    #create batch of training data
    batch_start = int(np.random.uniform(0, len(training_spins)-batch_size))
    spin_batch = model_spins[batch_start : batch_start+batch_size] #spin batch
    energy_batch = energies[batch_start : batch_start+batch_size] #energy batch
    
    #loop over states in batch
    for i in range(len(spin_batch)):
    	spin = spin_batch[i]
    
        #loop over all spins in state
    	for s in range(len(spin)):
    		accept = False #whether or not flip is accepted
    		spin[s] = spin[s] * (-1) #try flipping bit
    		flipped_energy = calc_energy(lambdas, spin) #energy of flipped state
    
    		if flipped_energy < energy_batch[i]: #accept if lower energy
    			accept = True
    		else: #if higher energy
    			p_rand = np.random.uniform(0, 1)
    			prob_accept = np.exp(-1*beta*(flipped_energy-energy_batch[i])) #probability of accepting
    			if p_rand < prob_accept: #accept if lower than probability
    				accept = True
    
            #if step is accepted
    		if accept==True:
    			energy_batch[i] = flipped_energy #update energy
    		else: #if not accepted
    			spin[s] = spin[s] * (-1) #flip bit back
    
    #add batches back to full sets
    model_spins[batch_start : batch_start+batch_size] = spin_batch
    energies[batch_start : batch_start+batch_size] = energy_batch
    
    #positive phase
    pos_phase=np.zeros(len_state)
    for i in range(len_state):
        pos_dotproduct = np.dot(training_spins.T[i],training_spins.T[(i+1) % len_state])
        pos_phase[i] = pos_dotproduct / len(training_spins)
    
    #negative phase
    neg_phase=np.zeros(len_state)
    for i in range(len_state):
        neg_dotproduct = np.dot(model_spins.T[i],model_spins.T[(i+1) % len_state])
        neg_phase[i] = neg_dotproduct / len(model_spins)

    #update lambdas
    lambdas += learning_rate * (pos_phase - neg_phase)

    if args.verbose: #if verbose
        #calculate model probabilities
        e_lambdas = [np.exp(-beta*calc_energy(lambdas, s)) for s in spin_states]
        z_lambda = np.sum(e_lambdas)
        p_lambdas = e_lambdas / z_lambda
        
        #calculate KL distance
        kl = KLdistance(prob_test, p_lambdas)
        kls.append(kl)
        
        #print KL distance occasionally
        if epoch % display_epoch == 0:
            print('epoch: '+str(epoch)+'/'+str(num_epoch)+'\t KL-distance: '+str(kl))
    
if args.verbose:
    #plot Kl distance
    plt.plot(np.arange(num_epoch), kls, 'k')
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('KL distance', fontsize=16)
    plt.savefig('KL-plot', bbox_inches='tight')

#output to file
out_f = open('out.txt', 'w')
out_str=''
for i in range(len_state):
    out_str += '('+str(i)+','+str((i+1)%len_state)+') : '+str(lambdas[i])+'\n'
out_f.write(out_str)
out_f.close()

