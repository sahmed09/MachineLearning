import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

stats.probplot(data, dist='norm', plot=plt)
plt.title('Normal Q-Q plot')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Ordered Values')
plt.grid(True)
plt.show()

n = 2000
observation = np.random.binomial(n=n, p=0.53, size=1000) / n

# standardize the observation
z = (observation - np.mean(observation)) / np.std(observation)

stats.probplot(z, dist='norm', plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# Generate a random sample from a normal distribution
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Generate a random sample from a right-skewed distribution (exponential distribution)
right_skewed_data = np.random.exponential(scale=1, size=1000)

# Generate a random sample from a left-skewed distribution (negative exponential distribution)
left_skewed_data = -np.random.exponential(scale=1, size=1000)

# Generate a random sample from an under-dispersed distribution (truncated normal distribution)
under_dispersed_data = np.random.normal(loc=0, scale=0.5, size=1000)
under_dispersed_data = under_dispersed_data[(under_dispersed_data > -1) & (under_dispersed_data < 1)]  # Truncate

# Generate a random sample from an over-dispersed distribution (mixture of normals)
over_dispersed_data = np.concatenate((np.random.normal(loc=-2, scale=1, size=500),
                                     np.random.normal(loc=2, scale=1, size=500)))

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
stats.probplot(normal_data, dist='norm', plot=plt)
plt.title('Q-Q Plot - Normal Distribution')

plt.subplot(2, 3, 2)
stats.probplot(right_skewed_data, dist='expon', plot=plt)
plt.title('Q-Q Plot - Right-skewed Distribution')

plt.subplot(2, 3, 3)
stats.probplot(left_skewed_data, dist='expon', plot=plt)
plt.title('Q-Q Plot - Left-skewed Distribution')

plt.subplot(2, 3, 4)
stats.probplot(under_dispersed_data, dist='norm', plot=plt)
plt.title('Q-Q Plot - Under-dispersed Distribution')

plt.subplot(2, 3, 5)
stats.probplot(over_dispersed_data, dist='norm', plot=plt)
plt.title('Q-Q Plot - Over-dispersed Distribution')

plt.tight_layout()
plt.show()

