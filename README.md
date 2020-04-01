# PHYS490Project
Recreation of results from the paper Self-Learning Monte Carlo Method for PHYS 490

There are three methods for generating states for the Ising model
- localudpate.py - the local update method
- wolffupdate.py - the global update method using the Wolff algorithm
- slmc.py - the self-learning method, using Wolff algoritm and trained parameters

To train the self-learning model, run
```
python train_slmc.py
```
The user can specify parameters at the top of the file.
- The J and K factors for the Hamiltonian
- The linear size of the state
- The number of steps in the Markov chain
- The temperature

This script will generate a specified number of states using the local update. The chain is initialized by a random state. After generating training states, the script will use linear regression to train the model parameters and output them.

The script will then generate a separate chain of states to be used for testing. The energy of the testing states will be calculated using the original Hamiltonian and the effective Hamiltonian. Then it will output the mean error between the two.

The script will also calculate the KL-distance for all 3 methods and plot them as a function of epoch. The script only does this for states that are 4x4 or smaller. The number of possible states increases with size as 2^(size * size), making this exceptionally slow for larger states. 


Depending on what the user wishes to do, there are three different scripts that out data or plots.

To run `main.py`, use
```
python main.py
```

Currently calculates Hamiltonian for a test state with both a nearest-neighbor and a plaquette term.

Step 1: Generates given number of states through local update. Takes a starting temperature to calculate thermodynamic beta and probabilities.

Step 2: Performs a linear regression to learn E0 and J1 in the effective Hamiltonian. Plots states from the original Hamiltonian and the linear fit.
