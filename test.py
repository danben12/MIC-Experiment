import pandas as pd
import numpy as np
<<<<<<< HEAD
import main
import statsmodels.api as sm




df = pd.read_csv('filtered_df2.csv', encoding='latin1')
centered=main.find_droplet_location(df)
# C1 = centered[centered['Well'] == 'C3']
bin_3 = df[df['log_Volume'] >= 3]
bin_3['laplacian_variance_normalized'] = (bin_3['laplacian_variance'] / bin_3['pixel_Area'])*(bin_3['Count']/bin_3['pixel_Area'])
bin_3 = bin_3[bin_3['laplacian_variance_normalized'] > 0]
bin_3['laplacian_variance_normalized'] = np.log10(bin_3['laplacian_variance_normalized'])
mean = bin_3['laplacian_variance_normalized'].mean()
std = bin_3['laplacian_variance_normalized'].std()
bin_3=bin_3[bin_3['laplacian_variance_normalized']<mean+-2*std]

# Find all the lines from bin_3 in df and change their Count values to NaN
df.loc[bin_3.index, 'Count'] = np.nan


def split_data_to_chips(df):
    chips = {}
    for chip in df['Well'].unique():
        chips[chip] = df[df['Well'] == chip]
    return chips


def log_mean_fill(df):
    epsilon = 1e-6
    chips = split_data_to_chips(df)
    for chip_key, chip_df in chips.items():
        for droplet in chip_df['Droplet'].unique():
            series = chip_df[chip_df['Droplet'] == droplet]['Count']
            result = series.copy()
            if pd.isna(result.iloc[0]):
                for idx in range(1, len(result)):
                    if pd.notna(result.iloc[idx]):
                        result.iloc[0] = result.iloc[idx]
                        break
            if pd.isna(result.iloc[-1]):
                for idx in range(len(result) - 2, -1, -1):
                    if pd.notna(result.iloc[idx]):
                        result.iloc[-1] = result.iloc[idx]
                        break

            for idx in range(1, len(series) - 1):
                if pd.isna(series.iloc[idx]):
                    before = series.iloc[idx - 1] if idx - 1 >= 0 else np.nan
                    after = series.iloc[idx + 1] if idx + 1 < len(series) else np.nan
                    before = before if before > 0 else epsilon
                    after = after if after > 0 else epsilon
                    if pd.notna(before) and pd.notna(after):
                        result.iloc[idx] = 10 ** ((np.log10(before) + np.log10(after)) / 2)
                    else:
                        found_valid_value = False
                        for j in range(1, idx + 1):
                            prev_value = series.iloc[idx - j] if idx - j >= 0 else np.nan
                            if pd.notna(prev_value) and prev_value > 0:
                                result.iloc[idx] = prev_value
                                found_valid_value = True
                                break
            result = result.round().astype(int)
            chip_df.loc[chip_df['Droplet'] == droplet, 'Count'] = result

    return pd.concat(chips.values())

def apply_lowess(df):
    chips = split_data_to_chips(df)
    for chip_key, chip_df in chips.items():
        for droplet in chip_df['Droplet'].unique():
            series = chip_df[chip_df['Droplet'] == droplet]['Count']
            result = series.copy()
            x = np.arange(len(result))
            log_counts = np.log10(result+1)
            lowess = sm.nonparametric.lowess(log_counts, x, frac=0.2)[:, 1]
            result = np.round(10 ** lowess - 1).astype(int)
            chip_df.loc[chip_df['Droplet'] == droplet, 'Count'] = result
    return pd.concat(chips.values())


# Example usage
filled_df = log_mean_fill(df)
filled_df=apply_lowess(filled_df)
filled_df.to_csv('filled_df.csv', index=False)




=======
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
>>>>>>> edb7d2d83efd1071c7b30013f76748f67164800e

