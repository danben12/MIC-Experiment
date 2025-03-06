import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example concentrations and response data
concentrations = np.array([0, 3.3, 10, 30])  # in µg/ml
response = np.array([0, 1, 5, 8])  # example response values (replace with actual data)

# Hill function
def hill_function(A, Vmax, K, n):
    return (Vmax * A**n) / (K**n + A**n)

# Fit the Hill function to the data
params, covariance = curve_fit(hill_function, concentrations, response, p0=[max(response), 10, 1])

# Extract parameters
Vmax, K, n = params
print(f"Estimated K: {K}, Vmax: {Vmax}, Hill coefficient: {n}")

# Plot the fit
A_fit = np.linspace(0, 30, 100)
response_fit = hill_function(A_fit, Vmax, K, n)

plt.plot(concentrations, response, 'o', label='Data')
plt.plot(A_fit, response_fit, '-', label='Fit')
plt.xlabel('Antibiotic Concentration (µg/ml)')
plt.ylabel('Response')
plt.legend()
plt.show()
