import numpy as np
import math
import glob
import os
import pandas as pd
from collections import Counter
from statsmodels.nonparametric.smoothers_lowess import lowess


def combine_csvs_with_roi_and_drop_errors(count_folder, roi_folder, errors_file=None):
    count_folder = os.path.join(count_folder, "Results_files", "Count")
    roi_folder = os.path.join(roi_folder, "Results_files", "ROIcsv")
    slice_folders = glob.glob(os.path.join(count_folder, '*'))
    print(f"Found {len(slice_folders)} slice folders.")

    results = []

    for slice_folder in slice_folders:
        slice_name = os.path.basename(slice_folder)
        print(f"Processing slice: {slice_name}")

        roi_file = os.path.join(roi_folder, f'{slice_name}.csv')

        if not os.path.exists(roi_file):
            print(f"ROI file for slice {slice_name} is missing.")
            continue

        roi_table = pd.read_csv(roi_file)
        roi_table = roi_table[[' ', 'Area', 'X', 'Y']]
        roi_table.columns = ['Droplet', 'Area', 'X', 'Y']

        count_files = glob.glob(os.path.join(slice_folder, '*.csv'))
        print(f"Found {len(count_files)} count files for slice {slice_name}.")

        if not count_files:
            print(f"No count files for slice {slice_name}.")
            continue

        for count_file in count_files:
            csv_name = os.path.basename(count_file).replace(".csv", "")
            temp = pd.read_csv(count_file)
            temp['Slice'] = csv_name  # Store the name of the CSV file under 'Slice'
            combined = pd.concat([temp, roi_table], axis=1)
            combined['Slice'] = csv_name  # Ensure 'Slice' column is present
            results.append(combined)

    if not results:
        print("No data to process.")
        return None

    all_results = pd.concat(results)
    all_results = all_results.drop(columns=['Count', 'Average Size', '%Area', 'Perim.'])
    all_results['Slice'] = all_results['Slice'].str.replace("_Simple Segmentation", "")
    all_results[['time', 'Well']] = all_results['Slice'].str.split('_', n=1, expand=True)
    all_results['time'] = all_results['time'].str.replace("h", "").astype(int)
    all_results['Slice'] = all_results['Slice'].str.replace(r'\d+h_', '', regex=True)
    all_results['InitialOD'] = 0.05

    if errors_file is not None:
        droplets_to_remove = pd.read_excel(errors_file)

        for slice_name in droplets_to_remove.columns:
            droplets_to_drop = droplets_to_remove[slice_name].dropna().astype(int).tolist()

            all_results = all_results.loc[
                          ~((all_results['Droplet'].isin(droplets_to_drop)) & (all_results['Slice'] == slice_name)), :]

    return all_results


# Usage example:
count_folder = r'C:\Users\Owner\Desktop\MIC experiment'
roi_folder = r'C:\Users\Owner\Desktop\MIC experiment'
# errors_file = 'Results_files/droplets_toRemove.xlsx'

combined_df = combine_csvs_with_roi_and_drop_errors(count_folder, roi_folder)
df = combined_df.copy()
df['DW'] = df['Droplet'].astype(str) + '_' + df['Slice'].astype(str)


def update_droplet_counts(df):
    # Group by 'DW' and 'Scenario', then calculate the sum of counts and any/all zeros condition
    group = df.groupby(['DW'])
    sum_counts = group['Total Area'].transform('sum')
    any_zeros = group['Total Area'].transform(lambda x: any(x == 0))
    all_zeros = group['Total Area'].transform(lambda x: all(x == 0))

    # Identify the droplets with sum counts < 25 and not all zeros
    update_condition = (sum_counts < 25) & ~all_zeros

    # Find 'DW's where any scenario meets the update condition
    dws_to_update = df.loc[update_condition, 'DW'].unique()

    # Update counts to 0 for both scenarios in these 'DW's
    df.loc[df['DW'].isin(dws_to_update), 'Total Area'] = 0

    return df


def remove_irregular_droplets(df):
    # Count the number of zeros in Weighted_Count for each DW
    zero_counts = df.groupby('DW')['Total Area'].transform(lambda x: (x == 0).sum())

    # Identify DWs with more than 4 but not all zeros in Weighted_Count
    more_than_5_zeros = zero_counts > 4
    all_zeros = df.groupby('DW')['Total Area'].transform(lambda x: all(x == 0))
    to_remove = more_than_5_zeros & ~all_zeros

    # Filter out the droplets that need to be removed
    filtered_df = df[~to_remove]

    # Create a DataFrame of removed droplets
    removed_droplets = df[to_remove].drop_duplicates()

    return filtered_df, removed_droplets


# Apply the function to the dataframe
filtered_df = update_droplet_counts(df.copy())

# Vectorized Area_to_Volume calculation with .loc
Theta = np.radians(32)
D = 2 * np.sqrt(filtered_df['Area'] / np.pi)
filtered_df.loc[:, 'Volume'] = ((np.pi * D ** 3) / 24) * (
            (2 - 3 * np.cos(Theta) + np.cos(Theta) ** 3) / (np.sin(Theta) ** 3))
filtered_df.loc[:, 'log_Volume'] = np.log10(filtered_df['Volume'])

# Binning
vol_labels = ['0 - 1','1 - 2','2 - 3','3 - 4', '4 - 5', '5 - 6', '6 - 7', '7 - 8']
cut_bins_vol = [0,1,2,3, 4, 5, 6, 7, 8]
filtered_df.loc[:, 'Bins_vol'] = pd.cut(filtered_df['log_Volume'], bins=cut_bins_vol)
filtered_df.loc[:, 'Bins_vol_txt'] = pd.cut(filtered_df['log_Volume'], bins=cut_bins_vol, labels=vol_labels)
filtered_df.to_csv('filtered_df2.csv', index=False)