# PHYS490Project
Recreation of results from the paper [Self-Learning Monte Carlo Method](https://arxiv.org/abs/1610.03137) for PHYS 490


There are three methods for generating states for the Ising model
- localudpate.py - the local update method
- wolffupdate.py - the global update method using the Wolff algorithm
- slmc.py - the self-learning method, using Wolff algoritm and trained parameters

# Training the model
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

Depending on what the user wishes to do, there are three different scripts that output data or plots.

# Plotting the magnetization
In order to generate a plot of magnetization as a function of temperature, run
```
python plot_magnetization.py
```
This plot is useful for determining the critical temperature. In addition to the parameters described above, the user can also specify
- The number of steps required for the chain to burn-in. These will be trimmed from the result
- The temperature range to plot over
- The mode the script will run in (0 for calculating and outputting data, 1 for plotting results)
- The linear sizes to plot (if mode is 1)
- The colors of the plots

This script will output a csv file containing the temperatures and mean magnetizations per site. By chaing the mode, these can be used later to plot several linear sizes as once to observe how size affects the magnetization.

# Comparing the autocorrelation
In order to calculate the autocorrelation for all 3 methods, run
In order to generate a plot of magnetization as a function of temperature, run
```
python plot_autocorr_compare.py
```
At the top of the file, the user can specify
- The J and K parameters of the Hamiltonian
- The temperature
- The size of the system
- The number of steps in the Markov chain for calculating the autocorrelation
- The number of steps in a separate Markov chain for training the SLMC
- The number of burn-in steps
- The maximum dt interval to be calculated for the autocorrelation
- The mode the script will run in (0 for calculating and outputting data, 1 for plotting results)

This script will ouput a csv file containing the dt intervals and the autocorrelation for all 3 methods. These can be used when changing the mode of the script to plot the autocorrelation. These files are also used in other scripts to fit an exponential function.

# Fitting a function to the autocorrelation
To fit an exponential function to the autocorrelation from the previous script, run
```
python fit_autotime.py
```
At the top of the file the user can specify
- J and K parameters for the Hamiltonian
- temperature

This will determine the autocorrelation time for each data set. These will be compiled at the end and plotted as a function of linear size. A power law is fit to this plot to determine the dependence on linear size for the autocorrelation time for each method. This plot will be saved and the fit parameters will be displayed.

# Folders
In the plots folder, one can find sample plots generated from the training script and analysis scripts.

In the data folder, one can find data files generated by the analysis scripts.

