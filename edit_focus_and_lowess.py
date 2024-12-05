import pandas as pd
import numpy as np
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
df.loc[bin_3.index, 'Count'] = np.nan
df['Count'] = df['Count'].apply(lambda x: np.nan if 0 < x < 8 else x)




def split_data_to_chips(df):
    chips = {}
    for chip in df['Well'].unique():
        chips[chip] = df[df['Well'] == chip]
    return chips

def replace_zero_with_nan(df):
    chips = split_data_to_chips(df)
    for chip_key, chip_df in chips.items():
        for droplet in chip_df['Droplet'].unique():
            series = chip_df[chip_df['Droplet'] == droplet]['Count']
            if series.iloc[0] == 0 and (series.iloc[1:6] != 0).any():
                chip_df.loc[chip_df[chip_df['Droplet'] == droplet].index[0], 'Count'] = np.nan
    return pd.concat(chips.values())


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


df=replace_zero_with_nan(df)
filled_df = log_mean_fill(df)
filled_df=apply_lowess(filled_df)
filled_df.to_csv('filled_df.csv', index=False)

