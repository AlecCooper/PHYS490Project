
#import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#global variables (temporary)
J_factor = 1.0e-4 #nearest-neighbor term factor
K_factor = 0.2e-4 #plaquette term factor
k_b = 8.617e-5 #Boltzmann constant in eV/K


def time_func(x, e, a):
    return a*np.power(x, e)

def autocorrelation_function(x, a, b, c, t):
    return a*np.exp(-t * (x - b)) + c

#plot autocorrelation and return autocorrelation time
def plot_autocorrelation(autocorr_chain, color=None):
    #perform fit
    plot_xs = np.arange(len(autocorr_chain))
    par, cov = curve_fit(autocorrelation_function, plot_xs, autocorr_chain, p0=[1.,0.,0.02,1./50.])
    #print(par)
    
    '''
    #calculate plot data points
    plot_ys=[]
    for x in plot_xs:
        y = autocorrelation_function(x, par[0], par[1], par[2], par[3])
        plot_ys.append(y)
        
    #plot fit
    if color is not None:
        plt.plot(plot_xs, plot_ys, color)
    else:
        plt.plot(plot_xs, plot_ys)
    '''
        
    #return autocorrelation time
    auto_time = 1./par[3]
    return auto_time




T = 3.3
sizes = [5,10,15,20,30,40,50,60,75]
wolff_sizes = [20,40,60,80,100]

times=[]
for i in range(len(sizes)):
    size = sizes[i]
    
    #read in local data
    try:
        in_data = np.genfromtxt('autocorrelation_time'+str(T)+'_'+str(size)+'.csv', delimiter=',')
    except:
        in_data = np.genfromtxt('autocorrelation_compare'+str(T)+'_'+str(size)+'.csv', delimiter=',')
    local_autocorr = in_data[1]

    #normalize
    local_autocorr /= max(local_autocorr)
    
    #perform autocorrelation fits
    local_time = plot_autocorrelation(local_autocorr)
    times.append(local_time)

    plt.plot(range(len(local_autocorr)), local_autocorr)
    
    #format plot and label
    plt.xlabel(r'$dt$', fontsize=16)
    plt.ylabel(r'$\langle M(t)M(t+dt) \rangle - \langle M \rangle ^2$', fontsize=16)
    
    #plt.show()
    plt.clf()

wolff_times=[]
slmc_times=[]
for i in range(len(wolff_sizes)):
    size = wolff_sizes[i]
    
    #read in wolff data
    in_data = np.genfromtxt('autocorrelation_time'+str(T)+'_'+str(size)+'_wolff.csv', delimiter=',')
    wolff_autocorr = in_data[1]

    #read in slmc data
    in_data = np.genfromtxt('autocorrelation_time'+str(T)+'_'+str(size)+'_slmc.csv', delimiter=',')
    slmc_autocorr = in_data[1]

    #normalize
    wolff_autocorr /= max(wolff_autocorr)
    slmc_autocorr /= max(slmc_autocorr)

    #perform autocorrelation fits
    wolff_time = plot_autocorrelation(wolff_autocorr)
    slmc_time = plot_autocorrelation(slmc_autocorr)
    wolff_times.append(wolff_time)
    slmc_times.append(slmc_time)

#plot autotimes
plt.plot(sizes, times, 'ko')
plt.plot(wolff_sizes, wolff_times, 'go')
plt.plot(wolff_sizes, slmc_times, 'bo')


#fit exponential
#local
local_par, cov = curve_fit(time_func, sizes, times)
local_err = np.sqrt(np.diag(cov))

step = (sizes[-1] - sizes[0])/100.
x_plot = np.arange(sizes[0], sizes[-1]+step, step)
y_plot = []
for x in x_plot:
    y_plot.append( time_func(x, local_par[0], local_par[1]) )

p1,=plt.plot(x_plot, y_plot, 'k')

#wolff
wolff_par, cov = curve_fit(time_func, wolff_sizes, wolff_times)
wolff_err = np.sqrt(np.diag(cov))

step = (wolff_sizes[-1] - wolff_sizes[0])/100.
x_plot = np.arange(wolff_sizes[0], wolff_sizes[-1]+step, step)
y_plot = []
for x in x_plot:
    y_plot.append( time_func(x, wolff_par[0], wolff_par[1]) )

p2,=plt.plot(x_plot, y_plot, 'g')

#slmc
slmc_par, cov = curve_fit(time_func, wolff_sizes, slmc_times)
slmc_err = np.sqrt(np.diag(cov))

y_plot = []
for x in x_plot:
    y_plot.append( time_func(x, slmc_par[0], slmc_par[1]) )

p3,=plt.plot(x_plot, y_plot, 'b')


#label and format
plt.legend([p1,p2,p3], ['local','wolff','slmc'])
plt.xlabel(r'$L$ - linear size', fontsize=16)
plt.ylabel(r'$\tau$', fontsize=16)
plt.savefig('autocorrelation_fit_all', bbox_inches='tight')
plt.clf()

#format and output fit parameters
ax = plt.gca()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)    

#add text to plot
exponent = '{0:.2f}'.format(local_par[0])+r'\pm'+'{0:.2f}'.format(local_err[0])
exponent = r'$L^{'+exponent+'}$'
amplitude = '{0:.2f}'.format(local_par[1])
plt.text(0.5,0.7,r'local: $\tau=$'+amplitude+' '+exponent,fontsize=24,ha='center',va='center',transform=ax.transAxes)
exponent = '{0:.2f}'.format(wolff_par[0])+r'\pm'+'{0:.2f}'.format(wolff_err[0])
exponent = r'$L^{'+exponent+'}$'
amplitude = '{0:.2f}'.format(wolff_par[1])
plt.text(0.5,0.5,r'wolff: $\tau=$'+amplitude+' '+exponent,fontsize=24,ha='center',va='center',transform=ax.transAxes)
exponent = '{0:.2f}'.format(slmc_par[0])+r'\pm'+'{0:.2f}'.format(slmc_err[0])
exponent = r'$L^{'+exponent+'}$'
amplitude = '{0:.2f}'.format(slmc_par[1])
plt.text(0.5,0.3,r'slmc: $\tau=$'+amplitude+' '+exponent,fontsize=24,ha='center',va='center',transform=ax.transAxes)

ax.set_axis_off()
plt.savefig('autocorrelation_parameters', bbox_inches='tight')


    