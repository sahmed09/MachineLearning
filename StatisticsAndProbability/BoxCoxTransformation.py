import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# To make this Example Reproducible
np.random.seed(0)

# Generate non-normal dataset (exponential)
original_data = np.random.exponential(size=1000)

# Performing Box-Cox Transformation on Non-Normal Dataset
transformed_data, best_lambda = stats.boxcox(original_data)

# Sub-Plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Plot the Distribution of Data Values
# sns.histplot(data=original_data, kde=True, ax=axes[0])
sns.histplot(data=original_data, kde=True, stat="density", linewidth=2, label="Non-Normal distribution", color="red",
             ax=axes[0])
axes[0].set_title('Non Normal Distribution')

# Plot the Distribution of the Transformed Data Values
# sns.histplot(data=transformed_data, kde=True, ax=axes[1])
sns.histplot(data=transformed_data, kde=True, stat="density", linewidth=2, label="Normal distribution", color="blue",
             ax=axes[1])
axes[1].set_title(f'Normal Distribution\nOptimal lambda: {best_lambda}')

axes[0].legend(loc="upper right")
axes[1].legend(loc="upper right")
fig.set_figheight(5)
fig.set_figwidth(10)
fig.tight_layout()
plt.show()
warnings.filterwarnings('ignore')

"""Another Approach"""

# To make this Example Reproducible
np.random.seed(0)

# Generate non-normal dataset (exponential)
original_data = np.random.exponential(size=1000)

# Performing Box-Cox Transformation on Non-Normal Dataset
transformed_data, best_lambda = stats.boxcox(original_data)

# Sub-Plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Plot the Distribution of Data Values
sns.distplot(original_data, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Non-Normal", color="green", ax=axes[0])

sns.distplot(transformed_data, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Normal", color="green", ax=axes[1])

# adding legends to the subplots
axes[0].legend(loc="upper right")
axes[1].legend(loc="upper right")
axes[1].set_title(f"Optimal lambda value: {best_lambda}")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
fig.tight_layout()
plt.show()

print(f"Lambda value used for Transformation: {best_lambda}")
