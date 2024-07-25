import numpy as np
from scipy.stats import norm
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Probability Density Function (PDF) of Normal Distribution
mu, sigma = 0, 1  # Mean and standard deviation
x = np.linspace(-5, 5, 1000)
prob_dens = norm.pdf(x, mu, sigma)

plt.plot(x, prob_dens, label='Normal Distribution')
plt.title('PDF of the Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Probability Mass Function (PMF) of Normal Distribution
mu, loc = 40, 10  # Mean and standard deviation
x = np.arange(0, 100, 1)
prob_mass = poisson.pmf(x, mu=mu, loc=loc)

plt.plot(x, prob_mass, label='Poisson Distribution')
plt.title('PMF of the Poisson Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend()
plt.show()
