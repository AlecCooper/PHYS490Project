# PHYS490Project
Recreation of results from the paper Self-Learning Monte Carlo Method for PHYS 490

To run `main.py`, use
```
python main.py
```

Currently calculates Hamiltonian for a test state with both a nearest-neighbor and a plaquette term.

Step 1: Generates given number of states through local update. Takes a starting temperature to calculate thermodynamic beta and probabilities.

Step 2: Performs a linear regression to learn E0 and J1 in the effective Hamiltonian. Plots states from the original Hamiltonian and the linear fit.
