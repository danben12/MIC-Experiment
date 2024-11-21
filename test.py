import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import main

df = pd.read_csv('filtered_df2.csv', encoding='latin1')
centered=main.find_droplet_location(df)
C1 = df[df['Slice'] == 'C1- Amp X10 MIC']
bin_3 = df[df['log_Volume'] >= 3]
bin_3['laplacian_variance_normalized'] = (bin_3['laplacian_variance'] / bin_3['pixel_Area'])*(df['Count']/df['pixel_Area'])
df = df[df['laplacian_variance_normalized'] > 0]
df['laplacian_variance_normalized'] = np.log10(df['laplacian_variance_normalized'])
mean = df['laplacian_variance_normalized'].mean()
std = df['laplacian_variance_normalized'].std()
df=df[df['laplacian_variance_normalized']<mean+-2*std]
print(len)
x = np.linspace(mean - 4*std, mean + 4*std, 1000)
y = norm.pdf(x, mean, std)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Bell Curve')
plt.hist(df['laplacian_variance_normalized'], bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
for i in range(1, 4):
    plt.axvline(mean + i*std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean - i*std, color='r', linestyle='dashed', linewidth=1)
    plt.text(mean + i*std, max(y)*0.8, f'+{i} std', color='r')
    plt.text(mean - i*std, max(y)*0.8, f'-{i} std', color='r')
plt.title('Bell Curve with Standard Deviation Lines')
plt.xlabel('Laplacian Variance Normalized (log10)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

