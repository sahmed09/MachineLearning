import numpy as np
from scipy.stats import norm, expon, binom, poisson, lognorm, bernoulli, chi2
import matplotlib.pyplot as plt

"""Continuous Distribution"""
# Normal or Gaussian Distribution
x = np.arange(-3, 3, 0.001)
prob_dens = norm.pdf(x)
plt.plot(x, prob_dens, label='Normal Distribution')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend()
plt.show()

# We specify that the mean value is 5.0, and the standard deviation is 2.0.
# Meaning that the values should be concentrated around 5.0, and rarely further away than 2.0 from the mean.
# And as you can see from the histogram, most values are between 4.0 and 6.0, with a top at approximately 5.0.
mu, sigma = 5.0, 2.0  # Mean and standard deviation
values = np.random.normal(mu, sigma, 10000)
plt.hist(values, bins=50, label='Normal Distribution')
plt.title('Normal Distribution')
plt.legend()
plt.show()

# Log-normal Distribution
x = np.linspace(0, 6, 1000)
stddev = 0.859455801705594
mean = 0.418749176686875
log_norm = lognorm(stddev, loc=mean)
log_norm = log_norm.pdf(x)
plt.plot(x, log_norm, label='Log-normal Distribution')
# plt.plot(x, lognorm.pdf(x, mean, stddev), label='Normal Distribution')
plt.title('Log-normal Distribution')
plt.xlabel('Values')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# mu, sigma = 3., 1.  # mean and standard deviation
# s = np.random.lognormal(mu, sigma, 1000)
# plt.hist(s, 100)
# plt.show()

# Exponential Distribution / Power Law
x = np.arange(0, 10, 0.001)
plt.plot(x, expon.pdf(x), label='Exponential Distribution')
plt.title('Exponential Distribution')
plt.xlabel('Values')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# Chi-square Distribution
x = np.arange(0, 20, 0.001)  # x-axis ranges from 0 to 20 with .001 steps
# define multiple Chi-square distributions
plt.plot(x, chi2.pdf(x, df=4), label='df: 4')  # Chi-square distribution with 4 degrees of freedom
plt.plot(x, chi2.pdf(x, df=8), label='df: 8')  # Chi-square distribution with 8 degrees of freedom
plt.plot(x, chi2.pdf(x, df=12), label='df: 12')  # Chi-square distribution with 12 degrees of freedom
plt.title('Chi-square Distribution')
plt.xlabel("Samples")
plt.ylabel("Density")
plt.legend()
plt.show()

"""Discrete Distribution"""
# Bernoulli Distribution
bd = bernoulli(0.8)  # Instance of Bernoulli distribution with parameter p = 0.8
x = [0, 1]  # Outcome of random variable either 0 or 1
plt.xlim(-2, 2)
plt.bar(x, bd.pmf(x), color='blue', label='Bernoulli Distribution')
plt.title('Bernoulli distribution (p=0.8)')
plt.xlabel('Values of random variable x (0, 1)')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Binomial Probability Mass Function
n, p = 10, 0.5
x = np.arange(0, 10, 0.001)
plt.plot(x, binom.pmf(x, n, p), label='Binomial Distribution')
plt.title('PMF of the Binomial Distribution')
plt.xlabel('Values')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Poisson Probability Mass Function
mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu), label='Poisson Distribution')
plt.title('PMF of the Poisson Distribution')
plt.xlabel('Values')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Uniform Distribution
values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, bins=50, label='Uniform Distribution')
plt.title('Uniform Distribution')
# plt.xlabel('Values')
# plt.ylabel('Probability')
plt.legend()
plt.show()
