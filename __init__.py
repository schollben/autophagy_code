import numpy as np

import matplotlib.pyplot as plt

# Generate random data
uniform_data = np.random.uniform(low=0, high=2, size=1000)  # Uniform distribution with mean around 1
poisson_data = np.random.poisson(lam=1, size=1000)  # Poisson distribution with lambda=1

# Create scatter plot
plt.scatter(uniform_data, poisson_data, alpha=0.5)
plt.title('Scatter Plot of Uniform vs Poisson Distribution')
plt.xlabel('Uniform Distribution')
plt.ylabel('Poisson Distribution')
plt.show()