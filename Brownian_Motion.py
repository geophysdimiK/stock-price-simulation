import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

#Function for calculating the temporal evolution of W
#for n=1000 timesteps
def wiener_process(dt=0.1, x0=0, n=1000):

    # Initialize W(t)
    W = np.zeros(n+1)

    # Create n+1 timesteps: t=0,1,2,3...n
    t = np.linspace(x0, n, n+1)

    # Use the cumulative sum for calculating W at every timestep
    W[1:n+1] = np.cumsum(npr.normal(0, np.sqrt(dt), n))

    return t, W

# Function for plotting W
def plot_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time(t)')
    plt.ylabel('W(t)')
    plt.title('Wiener process')
    plt.show()

[t, W] = wiener_process()

plot_process(t, W)

#Geometric Brownian Motion Simulation
def simulate_geometric_brownian_motion(S0, T=1, N=1000, mu=0.1, sigma=0.05):

    #timestep dt
    dt = T/N
    
    # Initialize W(t)
    W = np.zeros(N+1)
    
    # Create N timesteps
    t = np.linspace(0, T, N+1)
    
    # Use the cumulative sum for calculating W at every timestep
    W[1:N+1] = np.cumsum(npr.normal(0, np.sqrt(dt), N))
    
    #Calculate S(t)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    return t, S

def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel('Time (t)')
    plt.ylabel('Stock Price S(t)')
    plt.title('Geometric Brownian Motion')
    plt.show()

#Calculate S with an initial price S0 = 10 $
[t, S] = simulate_geometric_brownian_motion(10)

plot_simulation(t, S)

#Calculation of Geometric Brownian Motion using Aleatory
# Install aleatory, if necessary
#!pip install aleatory 

# Import the geometric Brownian motion class GBM
from aleatory.processes import GBM

# Create an instance of GBM
geometric_brownian_motion = GBM(drift=1, volatility=0.5, initial=1.0, T=1.0, rng=None)