import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.constants import value
from scipy.optimize import curve_fit
from scipy.special import kl_div
from sklearn.metrics import r2_score
import multiprocessing as mp
from scipy.stats import linregress

df= pd.read_csv(r'K:\21012025_BSF obj x10\final_data.csv',encoding='ISO-8859-1')
def generate_model(df):
    chips_data=df[df['Well'].isin(['C1','C5'])]
    grouped=chips_data.groupby('DW')
    log2_volume=lambda x: np.log2(x['Volume'].iloc[0])
    log2_volume=grouped.apply(log2_volume)
    density=lambda x: np.log2(x['Count'].iloc[0]/x['Volume'].iloc[0])
    density=grouped.apply(density)
    Rs=lambda x: np.log2(x['Count'].iloc[-4:].mean()/x['Count'].iloc[0])
    Rs=grouped.apply(Rs)
    data={
        'log2_volume':log2_volume,
        'density':density,
        'Rs':Rs
    }
    data=pd.DataFrame(data)
    data.replace([np.inf,-np.inf],np.nan,inplace=True)
    data.dropna(inplace=True)
    while True:
        X=data[['log2_volume','density']]
        y=data['Rs']
        X=sm.add_constant(X)
        model=sm.OLS(y,X).fit()
        data['residuals']=model.resid
        if data[(data['residuals'] > 2) | (data['residuals'] < -2)].empty:
            break
        data = data[(data['residuals'] <= 2) & (data['residuals'] >= -2)]
    return model

def preparing_dataset(df):
    grouped=df.groupby('DW')
    log2_volume=lambda x: np.log2(x['Volume'].iloc[0])
    log2_volume=grouped.apply(log2_volume)
    density=lambda x: np.log2(x['Count'].iloc[0]/x['Volume'].iloc[0])
    density=grouped.apply(density)
    N0=lambda x: x['Count'].iloc[0]
    N0=grouped.apply(N0)
    data={
        'log2_volume':log2_volume,
        'density':density,
        'N0':N0
    }
    data=pd.DataFrame(data)
    data.replace([np.inf,-np.inf],np.nan,inplace=True)
    data.dropna(inplace=True)
    return data

def predict_K(df):
    model=generate_model(df)
    data=preparing_dataset(df)
    X=data[['log2_volume','density']]
    X=sm.add_constant(X)
    data['Rs']=model.predict(X)
    data['K']=data['N0']*(2**data['Rs']).astype(int)
    return data

def calculated_K(df):
    grouped=df.groupby('DW')
    K=lambda x: x['Count'].iloc[-4:].mean()
    K=grouped.apply(K)
    return K

def generalized_logistic_model(t, N0, K,r, m):
    return K / ((1 - (1 - (K/N0)**m) * np.exp(-r * m * t))**(1/m))

def process_group(group):
    N = group['Count'].values
    t = group['time'].values
    k = N[-4:].mean()
    popt, _ = curve_fit(lambda t, r, m: generalized_logistic_model(t, N[0], k, r, m), t, N, maxfev=10000000)
    modeled_counts = generalized_logistic_model(t, N[0], k, popt[0], popt[1])
    if np.any(np.isnan(N)) or np.any(np.isnan(modeled_counts)):
        return None
    R2 = r2_score(N, modeled_counts)
    if R2 > 0.9:
        return popt[0], popt[1]
    return None

def median_r_and_m_values(df):
    chips_data = df[df['Well'].isin(['C1', 'C5'])]
    grouped = chips_data.groupby('DW')

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_group, [group for name, group in grouped])

    r_values = [result[0] for result in results if result is not None]
    m_values = [result[1] for result in results if result is not None]

    return np.median(r_values).astype(float), np.median(m_values).astype(float)

def simulate_predictions(predictions, r, m):
    t_values = np.linspace(0, 24, 25)  # Example time points
    simulation_results = []

    for index, row in predictions.iterrows():
        N0 = row['N0']
        K = row['K']
        simulated_counts = generalized_logistic_model(t_values, N0, K, r, m)
        simulation_results.append(simulated_counts.astype(int))  # Convert to int

    simulation_df = pd.DataFrame(simulation_results, columns=t_values)
    simulation_df.index = predictions.index  # Set the index to match predictions
    return simulation_df

def calc_slope_for_row(row):
    return row.rolling(window=3, min_periods=2, center=True).apply(
        lambda x: linregress(range(len(x)), x).slope)

def calc_slope(df):
    with mp.Pool(mp.cpu_count()) as pool:
        slopes_list = pool.map(calc_slope_for_row, [row for _, row in df.iterrows()])
    slopes = pd.DataFrame(slopes_list, index=df.index, columns=df.columns)
    return slopes

def hill_function(A, Vmax, K, n):
    return (Vmax * A**n) / (K**n + A**n)

def process_group(name, group, concentration):
    print(f"Processing group {name}")
    con = concentration[group['Well'].iloc[0]]
    if con == 0:
        return con, 0
    slope = group['Count'].rolling(window=3).apply(
        lambda x: linregress(range(len(x)), x).slope if len(x.dropna()) == 3 else np.nan
    )
    slope = slope.min()
    return con, slope

def calc_Kl(df):
    grouped = df.groupby('DW')
    concentration = {'C1': 0, 'C2': 30, 'C3': 10, 'C4': 3.3, 'C5': 0, 'C6': 3.3, 'C7': 30, 'C8': 10}
    data = {0: [], 3.3: [], 10: [], 30: []}

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(process_group, [(name,group, concentration) for name, group in grouped])

    for con, slope in results:
        data[con].append(slope)

    for key in data.keys():
        if key != 0:
            data[key] = [value for value in data[key] if value != 0]
        data[key] = np.mean(data[key])

    Vmax = min(data.values())
    x_data=np.array(list(data.keys()))
    y_data=np.array(list(data.values()))

    params, covariance = curve_fit(lambda A, K, n: hill_function(A, Vmax, K, n), x_data, y_data)
    K, n = params
    plt.scatter(x_data, y_data)
    plt.xlabel('Concentration')
    plt.ylabel('Slope')
    plt.title(f'Kl: {K:.2f}, h: {n:.2f}')
    A_fit = np.linspace(0, 30, 100)
    response_fit = hill_function(A_fit, Vmax, K, n)
    plt.plot(A_fit, response_fit)
    plt.axvline(x=K, color='r', linestyle='--', label=f'Kl: {K:.2f}')
    y_kl = hill_function(K, Vmax, K, n)
    plt.axhline(y=y_kl, color='b', linestyle='--', label=f'y at Kl: {y_kl:.2f}')
    plt.legend()
    plt.show()


# def growth_lysis_model(df,slopes,Kl,n):
#     grouped=df.groupby('DW')
#     t=np.linspace(0,24,25)
#     concentration = {'C1': 0, 'C2': 30, 'C3': 10, 'C4': 3.3, 'C5': 0, 'C6': 3.3, 'C7': 30, 'C8': 10}
#     for index, row in slopes.iterrows():
#         A=concentration[row['Well']]
#         if A==0:
#             Vmax=0
#         else:
#             Vmax=row.min()
#         g=row[t]
#         if A<Kl:
#             l=0
#         else:
#             l=(Vmax*(A**n)/(Kl**n+A**n))*g
#         model=(g-l)*grouped



if __name__ == '__main__':
    Kl, n = calc_Kl(df)
    # predictions=predict_K(df)
    # r,m=median_r_and_m_values(df)
    # simulation_df = simulate_predictions(predictions, r, m)
    # slopes = calc_slope(simulation_df)




