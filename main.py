import math
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from bokeh.io import output_file
from bokeh.models import Div, HoverTool, TapTool, ColorBar, CheckboxGroup, LogTicker, LinearColorMapper, \
    BasicTicker, Legend, LegendItem, BoxAnnotation, Span, Whisker, Label, Spacer, \
    ColumnDataSource, CDSView, Select, CustomJS, GlyphRenderer
from bokeh.layouts import column, row
from bokeh.palettes import Category20, RGB, bokeh
from bokeh.plotting import figure, show, markers
from networkx.algorithms.bipartite import density
from scipy.stats import linregress
from matplotlib import cm
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde





def read_data():
    Tk().withdraw()
    file_path = askopenfilename()
    if file_path:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    else:
        return None
def split_data_to_chips():
    data=read_data()
    chips = {slice_id: df.reset_index(drop=True) for slice_id, df in data.groupby('Slice')}
    return chips
def initial_stats(data):
    filtered_chips = {slice_id: df[df['time'] == 0].reset_index(drop=True) for slice_id, df in data.items()}
    return filtered_chips
def get_slice(data, slice_id):
    slice_data = data[slice_id]
    experiment_time = slice_data['time'].max()
    time_steps = np.diff(sorted(slice_data['time'].unique()))[0]
    chip = {droplet_id: df.reset_index(drop=True) for droplet_id, df in slice_data.groupby('Droplet')}
    return chip,experiment_time,time_steps

def stats_box(df,time, max_step,chip_name):
    volume =np.log10(df['Volume'].sum())
    mean=np.log10(df['Volume'].mean())
    std=np.log10(df['Volume'].std())
    bacteria_pool = df['Count'].sum()
    chip_density = np.log10(bacteria_pool / 10**volume)

    stats_text = (f"Chip: {chip_name}<br>"
                  f"Total droplets volume: 10<sup>{volume:.2f}</sup><br>"
                  f"Actual Droplets Mean Size: 10<sup>{mean:.2f}</sup><br>"
                  f"Actual Droplets Standard Deviation: 10<sup>{std:.2f}</sup><br>"
                  f"Number of bacteria: {bacteria_pool}<br>"
                  f"Chip Density: 10<sup>{chip_density:.2f}</sup><br>"
                  f"Time: {time}<br>"
                  f"Time Step: {max_step}<br>")
    stats_div = Div(
        text=stats_text,
        width=500,
        height=300
    )
    stats_div.styles = {
        'text-align': 'left',
        'margin': '10px auto',
        'font-size': '12pt',
        'font-family': 'Arial, sans-serif',
        'color': 'black',
        'background-color': 'lightgray',
        'border': '1px solid black',
        'padding': '20px',
        'box-shadow': '5px 5px 5px 0px lightgray',
        'border-radius': '10px',
        'line-height': '1.5em',
        'font-weight': 'bold',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',
        'overflow-wrap': 'break-word',
        'text-overflow': 'ellipsis',
        'hyphens': 'auto'
    }
    return column(stats_div)

def droplet_histogram(df):
    bins = np.logspace(3, 8, num=16)
    hist = figure(title='Histogram of Droplet Size', x_axis_type='log',
                  x_axis_label='Volume', y_axis_label='Frequency', output_backend="webgl")
    hist_data = np.histogram(df['Volume'], bins=bins)
    hist_data_occupied = np.histogram(df[df['Count'] > 0]['Volume'], bins=bins)
    # Create a ColumnDataSource from histogram data
    source = ColumnDataSource(data=dict(
        top=hist_data[0],
        bottom=np.zeros_like(hist_data[0]),
        left=bins[:-1],
        right=bins[1:],
        top_occupied=hist_data_occupied[0]
    ))
    view = CDSView()
    hist.quad(top='top', bottom='bottom', left='left', right='right',
              color='gray', alpha=0.5, legend_label='Volume', source=source, view=view)
    hist.quad(top='top_occupied', bottom='bottom', left='left', right='right',
              color='blue', alpha=0.1, legend_label='occupied droplets', source=source, view=view)
    droplet_num = int(df['Volume'].count())
    occupied_droplets = int(df[df['Count'] > 0]['Volume'].count())
    occupancy_rate = occupied_droplets / droplet_num
    stats_text = f"Droplets count: {droplet_num}<br>Occupied Droplets: {occupied_droplets}<br>Occupancy Rate: {occupancy_rate:.2%}"
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(hist, stats_div)
    return combined_plot


def N0_Vs_Volume(df, Vc):
    source = ColumnDataSource(df)
    view = CDSView()
    scatter = figure(title='N0 vs. Volume', x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='N0', output_backend="webgl")
    scatter_renderer = scatter.scatter('Volume', 'Count', source=source, view=view, color='gray', alpha=1,
                                       legend_label='N0 vs. Volume')
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Count', '@Count'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter_renderer])
    scatter.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    filtered_df = df[df['Count'] > 0]
    filtered_df = filtered_df[filtered_df['Volume'] >= Vc]
    x = np.log10(filtered_df['Volume'])
    y = np.log10(filtered_df['Count'])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_values = np.linspace(min(df['Volume']), max(df['Volume']), 100)
    y_values = 10 ** (intercept + slope * np.log10(x_values))
    stats_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R² value: {r_value ** 2:.2f}'
    scatter.line(x_values, y_values, color='red', legend_label='Linear Regression')
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(scatter, stats_div)
    return combined_plot


def Initial_Density_Vs_Volume(df, initial_density):
    df['initial density'] = df['Count'] / df['Volume']
    source = ColumnDataSource(df)
    view = CDSView()
    scatter = figure(title='Initial Density vs. Volume', x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='Initial Density', output_backend="webgl")
    scatter_renderer = scatter.scatter('Volume', 'initial density', source=source, view=view, color='gray', alpha=1)
    hover = HoverTool(
        tooltips=[('Volume', '@Volume'), ('Initial Density', '@{initial density}'), ('Droplet ID', '@Droplet')],
        renderers=[scatter_renderer])
    scatter.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    filtered_sorted_df = df[df['initial density'] > 0].sort_values(by='Volume').reset_index()
    log_density = np.log10(filtered_sorted_df['initial density'])
    rolling_mean = log_density.rolling(window=100,min_periods=1).mean()
    scatter.line(filtered_sorted_df['Volume'], 10 ** rolling_mean, color='red')
    scatter.line([min(df['Volume']), max(df['Volume'])], [initial_density, initial_density], color='green')
    convergence_window = 2
    tolerance = 0.05
    differences = np.abs(1 - (10 ** rolling_mean / initial_density))
    for i in range(len(differences) - convergence_window):
        window_mean_diff = differences.iloc[i:i + convergence_window].mean()
        if window_mean_diff <= tolerance:
            closest_index = i + convergence_window // 2
            break
        else:
            closest_index = differences.idxmin()
    closest_point = filtered_sorted_df.loc[closest_index]
    closest_volume = closest_point['Volume']
    vline = Span(location=closest_volume, dimension='height', line_color='blue', line_dash='dashed', line_width=2)
    scatter.add_layout(vline)
    scatter.renderers.append(vline)
    invisible_line = scatter.line([0], [0], color='blue', line_dash='dashed', line_width=2)
    legend = Legend(items=[
        LegendItem(label='Initial density vs. Volume', renderers=[scatter.renderers[0]]),
        LegendItem(label='Rolling Mean', renderers=[scatter.renderers[1]]),
        LegendItem(label='Initial Density', renderers=[scatter.renderers[2]]),
        LegendItem(label='Vc', renderers=[invisible_line])
    ], location='top_right')
    scatter.add_layout(legend)
    return scatter, closest_volume


def Fraction_in_each_bin(dic, time):
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[(combined_df['time'] == 0) | (combined_df['time'] == time)]
    combined_df['bottom_bin'] = np.log10(combined_df['Volume']).apply(math.floor)
    combined_df['top_bin'] = np.log10(combined_df['Volume']).apply(math.ceil)
    start_total = combined_df[combined_df['time'] == 0]['Count'].sum()
    end_total = combined_df[combined_df['time'] == time]['Count'].sum()
    bins = combined_df.groupby(['bottom_bin', 'top_bin', 'time'])['Count'].sum()
    bins = bins.unstack().fillna(0)
    bins['start fraction'] = bins[0] / start_total * 100
    bins['end fraction'] = bins[time] / end_total * 100
    p = figure(title='Fraction of Population in Each Bin at Start and End of Simulation', x_axis_label='Bin Range', y_axis_label='Fraction of Population', output_backend="webgl",y_range=(0,100))
    bin_centers = [(start_bin + end_bin) / 2 for start_bin, end_bin in bins.index]
    bin_width = 0.4
    p.vbar(x=[bin_center - bin_width / 2 for bin_center in bin_centers], top=bins['start fraction'], width=bin_width, color='blue', legend_label='Start')
    p.vbar(x=[bin_center + bin_width / 2 for bin_center in bin_centers], top=bins['end fraction'], width=bin_width, color='green', legend_label='End')
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p

def fold_change(data_dict,Vc):
    fold_change = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])  # Array to store Google Drive URLs
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    df = pd.DataFrame({'Volume': Volume, 'fold change': fold_change, 'Droplet': droplet_id, 'Google Drive Link': google_drive_url})
    df = df.sort_values(by='Volume').reset_index(drop=True)
    sub_df = df[df['fold change'] > min_fc].reset_index(drop=True)
    sub_df['moving average'] = sub_df['fold change'].rolling(window=100,min_periods=1).mean()
    source = ColumnDataSource(df)
    sub_source = ColumnDataSource(sub_df)
    view = CDSView()
    sub_view = CDSView()
    scatter = figure(title='Volume vs. Log2 Fold Change', x_axis_type='log', y_axis_type='linear',
                     x_axis_label='Volume', y_axis_label='Log2 Fold Change', output_backend="webgl",width=900, height=600)
    scatter.scatter('Volume', 'fold change', source=source, view=view, color='gray', alpha=1)
    scatter.line('Volume', 'moving average', source=sub_source, view=sub_view, color='red')
    total_initial_counts=sum(value['Count'].iloc[0] for value in data_dict.values() if value['Count'].iloc[0] != 0)
    total_final_counts=sum(value['Count'].iloc[-4:].mean() for value in data_dict.values() if value['Count'].iloc[0] != 0)
    metapopulation_fold_change = np.log2(total_final_counts / total_initial_counts)
    scatter.line([min(df['Volume']), max(df['Volume'])], [metapopulation_fold_change, metapopulation_fold_change], color='green')
    scatter.line([Vc,Vc],[df['fold change'].min(),df['fold change'].max()],color='blue',line_dash='dashed',line_width=2)
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Fold Change', '@{fold change}'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter.renderers[0]])
    scatter.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    legend = Legend(items=[
        LegendItem(label='Fold Change', renderers=[scatter.renderers[0]]),
        LegendItem(label='Moving Average', renderers=[scatter.renderers[1]]),
        LegendItem(label='Metapopulation Fold Change', renderers=[scatter.renderers[2]]),
        LegendItem(label='Vc', renderers=[scatter.renderers[3]])
    ], location='top_right')
    scatter.add_layout(legend, 'right')
    return scatter

def growth_curves(dict):
    valid_droplets=[]
    for key, value in dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df=pd.concat(valid_droplets,ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE']=result['std']/np.sqrt(result['sample_count'])
    result['mean + se'] = result['mean'] + result['SE']
    result['mean - se'] = result['mean'] - result['SE']

    metapopulation = df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    p1 = figure(title='Growth Curves', x_axis_label='Time', y_axis_label='Mean Count', width=800, height=600, output_backend="webgl")
    p2 = figure(title='Growth Log Scale', x_axis_label='Time', y_axis_label='Mean Count', width=800, height=600, output_backend="webgl", y_axis_type='log')
    colors = Category20[20]
    color_index = 0
    legend_items_1 = []
    legend_items_2 = []
    for (lower_bin, upper_bin), group in result.groupby(['lower bin', 'upper bin']):
        source = ColumnDataSource(group)
        line_1 = p1.line('time', 'mean', source=source, color=colors[color_index], line_width=2)
        varea_1 = p1.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=colors[color_index],
                           alpha=0.2)
        line_2 = p2.line('time', 'mean', source=source, color=colors[color_index], line_width=2)
        varea_2 = p2.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=colors[color_index],
                           alpha=0.2)
        legend_item_1 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_1, varea_1])
        legend_items_1.append(legend_item_1)
        legend_item_2 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_2, varea_2])
        legend_items_2.append(legend_item_2)
        color_index = (color_index + 1) % len(colors)
    metapopulation_source = ColumnDataSource(metapopulation)
    metapopulation_line_1 = p1.line('time', 'metapopulation', source=metapopulation_source, color='black', line_width=3)
    metapopulation_line_2 = p2.line('time', 'metapopulation', source=metapopulation_source, color='black', line_width=3)
    legend_items_1.append(LegendItem(label='Metapopulation', renderers=[metapopulation_line_1]))
    legend_items_2.append(LegendItem(label='Metapopulation', renderers=[metapopulation_line_2]))
    legend_1 = Legend(items=legend_items_1, location='top_right')
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p1.add_layout(legend_1, 'right')
    p1.legend.click_policy = 'hide'
    p2.add_layout(legend_2, 'right')
    p2.legend.click_policy = 'hide'
    return row(p1, p2)

def normalize_growth_curves(data_dict):
    valid_droplets = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])
    result['max_mean'] = result.groupby(['lower bin', 'upper bin'])['mean'].transform('max')
    result['normalized_mean'] = result['mean'] / result['max_mean']
    result['normalized_SE'] = result['SE'] / result['max_mean']
    result['mean + SE'] = result['normalized_mean'] + (result['normalized_SE'])
    result['mean - SE'] = result['normalized_mean'] - (result['normalized_SE'])
    p1 = figure(title='Normalized Growth Curves', x_axis_label='Time', y_axis_label='Normalized Mean Count', width=800, height=600, output_backend="webgl")
    p2 = figure(title='Normalized Growth Log Scale', x_axis_label='Time', y_axis_label='Normalized Mean Count', width=800, height=600, output_backend="webgl", y_axis_type='log')
    colors = Category20[20]
    color_index = 0
    legend_items_1 = []
    legend_items_2 = []
    for (lower_bin, upper_bin), group in result.groupby(['lower bin', 'upper bin']):
        source = ColumnDataSource(group)
        line_1 = p1.line('time', 'normalized_mean', source=source, color=colors[color_index], line_width=2)
        varea_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=colors[color_index],
                            alpha=0.2)
        line_2 = p2.line('time', 'normalized_mean', source=source, color=colors[color_index], line_width=2)
        varea_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=colors[color_index],
                            alpha=0.2)
        legend_item_1 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_1, varea_1])
        legend_items_1.append(legend_item_1)
        legend_item_2 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_2, varea_2])
        legend_items_2.append(legend_item_2)
        color_index = (color_index + 1) % len(colors)
    legend_1 = Legend(items=legend_items_1, location='top_right')
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p1.add_layout(legend_1, 'right')
    p1.legend.click_policy = 'hide'
    p2.add_layout(legend_2, 'right')
    p2.legend.click_policy = 'hide'
    return row(p1, p2)

def last_4_hours_average(chip, volume):
    last_4_hours = {droplet_id: df[df['time'] > 20].reset_index(drop=True) for droplet_id, df in chip.items()}
    average_counts = np.array([df['Count'].mean() for df in last_4_hours.values()])
    min_average_count = 0.1
    average_counts = np.where(average_counts == 0, min_average_count, average_counts)
    droplet_sizes = [df['Volume'].iloc[0] for df in chip.values()]
    droplet_ids = [df['Droplet'].iloc[0] for df in chip.values()]
    google_drive_urls = [df['Google Drive Link'].iloc[0] for df in chip.values()]  # Add URL
    data = pd.DataFrame({'Volume': droplet_sizes, 'Average Count': average_counts, 'Droplet': droplet_ids, 'Google Drive Link': google_drive_urls})
    data = data.sort_values(by='Volume').reset_index(drop=True)
    data_before = data[data['Volume'] <= volume]
    data_after = data[data['Volume'] > volume]
    data_before = data_before[data_before['Average Count'] > data_before['Average Count'].min()]
    data_after = data_after[data_after['Average Count'] > data_after['Average Count'].min()]
    if not data_before.empty:
        slope_before, intercept_before, r_value_before, _, _ = linregress(np.log10(data_before['Volume']), np.log10(data_before['Average Count']))
        x_values_before = np.linspace(data_before['Volume'].min(), volume, 100)
        y_values_before = 10 ** (intercept_before + slope_before * np.log10(x_values_before))
    else:
        x_values_before, y_values_before = np.array([]), np.array([])
        slope_before, r_squared_before = None, None

    if not data_after.empty:
        slope_after, intercept_after, r_value_after, _, _ = linregress(np.log10(data_after['Volume']), np.log10(data_after['Average Count']))
        x_values_after = np.linspace(volume, data_after['Volume'].max(), 100)
        y_values_after = 10 ** (intercept_after + slope_after * np.log10(x_values_after))
    else:
        x_values_after, y_values_after = np.array([]), np.array([])
        slope_after, r_squared_after = None, None

    source = ColumnDataSource(data)
    view = CDSView()
    scatter = figure(title='Average Number of Bacteria in Last 4 Hours vs. Droplet Size', x_axis_type='log',
                     y_axis_type='log', x_axis_label='Volume', y_axis_label='Average Count', output_backend="webgl", width=900, height=600)
    scatter.scatter('Volume', 'Average Count', source=source, view=view, color='gray', alpha=1)
    regression_before_renderer = None
    regression_after_renderer = None
    if x_values_before.any() and y_values_before.any():
        regression_before_renderer = scatter.line(x_values_before, y_values_before, color='red')
        label_x_before = (x_values_before[0] + x_values_before[-1]) / 2
        label_y_before = 10 ** (intercept_before + slope_before * np.log10(label_x_before))
        label_before = Label(x=label_x_before, y=label_y_before, text=f'Slope: {slope_before:.2f}', text_color='red')
        scatter.renderers.append(label_before)
        scatter.add_layout(label_before)
    if x_values_after.any() and y_values_after.any():
        regression_after_renderer = scatter.line(x_values_after, y_values_after, color='blue')
        label_x_after = (x_values_after[0] + x_values_after[-1]) / 2
        label_y_after = 10 ** (intercept_after + slope_after * np.log10(label_x_after))
        label_after = Label(x=label_x_after, y=label_y_after, text=f'Slope: {slope_after:.2f}', text_color='blue')
        scatter.renderers.append(label_after)
        scatter.add_layout(label_after)
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Average Count', '@{Average Count}'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter.renderers[0]])
    scatter.add_tools(hover)
    vline = Span(location=volume, dimension='height', line_color='blue', line_dash='dashed', line_width=2)
    scatter.add_layout(vline)
    scatter.renderers.append(vline)
    invisible_line = scatter.line([0], [0], color='blue', line_dash='dashed', line_width=2)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    legend_items = [
        LegendItem(label='Average Count', renderers=[scatter.renderers[0]]),
        LegendItem(label='Vc', renderers=[invisible_line])
    ]
    if regression_before_renderer:
        legend_items.append(LegendItem(label='Regression Before', renderers=[regression_before_renderer]))
    if regression_after_renderer:
        legend_items.append(LegendItem(label='Regression After', renderers=[regression_after_renderer]))
    legend = Legend(items=legend_items, location='top_right')
    scatter.add_layout(legend, 'right')
    return scatter

def find_droplet_location(df):
    square_size = 8110
    circle_radius = square_size / 2
    circle_center_x = square_size / 2
    circle_center_y = square_size / 2
    df['distance_to_center'] = np.sqrt((df['X'] - circle_center_x) ** 2 + (df['Y'] - circle_center_y) ** 2)
    df['is_inside_circle'] = df['distance_to_center'] <= circle_radius
    return df[df['is_inside_circle']].reset_index(drop=True)


def death_rate_by_bins(dict):
    valid_droplets = []
    for key, value in dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])['Count'].sum().reset_index(name='Count')
    mask = grouped['Count'] > 0
    grouped['log_count'] = grouped[mask]['Count'].apply(np.log)
    window_size = 4
    grouped['slope'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[0]))
    grouped['standard_error'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[4]))
    grouped['slope - standard_error'] = grouped['slope'] - grouped['standard_error']
    grouped['slope + standard_error'] = grouped['slope'] + grouped['standard_error']

    # Calculate metapopulation death rate
    metapopulation = df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0])
    p = figure(title='Death Rate by Bins', x_axis_label='Time', y_axis_label='Slope', width=800, height=600,
               output_backend="webgl")
    colors = Category20[20]
    color_index = 0
    legend_items = []
    for (lower_bin, upper_bin), group in grouped.groupby(['lower bin', 'upper bin']):
        source = ColumnDataSource(group)
        view = CDSView()
        line = p.line('time', 'slope', source=source, view=view, color=colors[color_index], line_width=2)
        varea = p.varea(x='time', y1='slope - standard_error', y2='slope + standard_error', source=source,
                        color=colors[color_index], alpha=0.2)
        legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line, varea])
        legend_items.append(legend_item)
        color_index = (color_index + 1) % len(colors)

    # Add metapopulation death rate line
    metapopulation_source = ColumnDataSource(metapopulation)
    metapopulation_line = p.line('time', 'slope', source=metapopulation_source, line_width=3, color='black')
    legend_items.append(LegendItem(label='Metapopulation Death Rate', renderers=[metapopulation_line]))

    legend = Legend(items=legend_items, location='top_right')
    p.add_layout(legend, 'right')
    p.legend.click_policy = 'hide'
    return p


def death_rate_by_droplets(data_dict,chip):
    volumes = []
    max_death_rate = []
    droplet_ids = []
    google_drive_urls = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log10)
            value['slope'] = value['log_count'].rolling(window_size).apply(
                lambda x: linregress(range(window_size), x)[0])
            volumes.append(value['Volume'].iloc[0])
            if chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
                max_death_rate.append(value['slope'].max())
            else:
                max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
            google_drive_urls.append(value['Google Drive Link'].iloc[0])
    df = pd.DataFrame({
        'Volume': np.log10(volumes),
        'Slope': max_death_rate,
        'Droplet': droplet_ids,
        'Google Drive Link': google_drive_urls
    })
    df['upper bin'] = df['Volume'].apply(math.ceil)
    df['lower bin'] = df['Volume'].apply(math.floor)
    grouped = df.groupby(['lower bin', 'upper bin'])
    colors = Category20[20]
    color_map = {group: colors[i % len(colors)] for i, group in enumerate(grouped.groups.keys())}
    df['color'] = df.apply(lambda row: color_map[(row['lower bin'], row['upper bin'])], axis=1)
    valid_droplets = [value for key, value in data_dict.items() if value['Count'].iloc[0] != 0]
    metapopulation_df = pd.concat(valid_droplets, ignore_index=True)
    metapopulation = metapopulation_df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    window_size = 4
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0]
    )
    if chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
        mean_death_rate = metapopulation['slope'].max()
        title = 'Maximal slope by Droplets'
    else:
        mean_death_rate = metapopulation['slope'].min()
        title = 'Minimal slope by Droplets'
    p = figure(title=title, x_axis_label='log 10 Volume', y_axis_label='slope',output_backend="webgl",width=800, height=600)
    permutations_data = pd.DataFrame(columns=['first compared bin', 'second compared bin','p-value'])
    lower_bin=df['lower bin'].unique()
    for i in lower_bin:
        for j in lower_bin:
            if i<j:
                data1 = df[df['lower bin'] == i]['Slope']
                data2 = df[df['lower bin'] == j]['Slope']
                if len(data1) >= 2 and len(data2) >= 2:
                    p_value = permutation_test((data1, data2),lambda x, y: np.mean(x) - np.mean(y), n_resamples=1000, alternative='two-sided').pvalue
                    permutations_data=pd.concat([permutations_data,pd.DataFrame({'first compared bin':i,'second compared bin':j,'p-value':p_value},index=[0])]).reset_index(drop=True)
    adjusted_results = multipletests(permutations_data['p-value'], method='fdr_bh')
    permutations_data['adjusted p-value'] = adjusted_results[1]
    def replace_p_values(value):
        if value > 0.05:
            return 'NS'
        elif 0.05 >= value > 0.01:
            return '*'
        elif 0.01 >= value > 0.001:
            return '**'
        else:
            return '***'
    permutations_data['adjusted p-value'] = permutations_data['adjusted p-value'].apply(replace_p_values)
    for index, row in permutations_data.iterrows():
        first_bin = row['first compared bin']
        second_bin = row['second compared bin']
        p_value_label = row['adjusted p-value']
        y_position = df['Slope'].max() + (index + 1) * 0.1
        p.line(x=[first_bin, second_bin+1], y=[y_position, y_position], line_color="black")
        p.line(x=[first_bin, first_bin], y=[y_position, y_position -0.02], line_color="black")
        p.line(x=[second_bin+1, second_bin+1], y=[y_position, y_position -0.02], line_color="black")
        label = Label(x=(first_bin + second_bin+1) / 2, y=y_position, text=p_value_label, text_color="red",
                      text_align="center")
        p.add_layout(label)
    source = ColumnDataSource(df)
    scatter = p.scatter(x='Volume', y='Slope', source=source, color='color')
    num_points = len(scatter.data_source.data['Volume'])
    for (lower_bin, upper_bin), group in grouped:
        color = color_map[(lower_bin, upper_bin)]
        q1 = group['Slope'].quantile(0.25)
        q3 = group['Slope'].quantile(0.75)
        median = group['Slope'].median()
        iqr = q3 - q1
        upper_whisker = min(group['Slope'].max(), q3 + 1.5 * iqr)
        lower_whisker = max(group['Slope'].min(), q1 - 1.5 * iqr)
        p.quad(top=[q3], bottom=[q1], left=[lower_bin], right=[upper_bin], fill_color=color, alpha=0.3)
        p.segment(x0=[lower_bin], y0=[median], x1=[upper_bin], y1=[median], line_color="black")
        p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[upper_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q3],
                  line_color="black")
        p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[lower_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q1],
                  line_color="black")
        p.line(x=[lower_bin, upper_bin], y=[upper_whisker, upper_whisker], line_color="black")
        p.line(x=[lower_bin, upper_bin], y=[lower_whisker, lower_whisker], line_color="black")
    p.line(x=[3,8], y=[mean_death_rate, mean_death_rate],
           line_dash='dashed', line_color='black', line_width=2)
    legend = Legend(items=[LegendItem(label='Metapopulation Slope', renderers=[p.renderers[-1]])], location='top_right')
    p.add_layout(legend, 'right')
    tap_tool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const url = source.data['Google Drive Link'][selected_index];
            window.open(url, "_blank");

            // Highlight the selected point and shade others
            for (let i = 0; i < source.data['alpha'].length; i++) {
                source.data['alpha'][i] = (i === selected_index) ? 1.0 : 0.1;
            }
            source.change.emit();
        }
    """))
    p.add_tools(tap_tool)
    hover = HoverTool(
        tooltips=[
            ('Log 10 Volume', '@Volume'),
            ('Slope', '@{Slope}'),
            ('Droplet', '@Droplet')
        ],
        renderers=[scatter]
    )
    p.add_tools(hover)
    return p

def distance_Vs_Volume_histogram(df):
    df = df.copy()
    distance_bins = [0, 1000, 2000, 3000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4055"]
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))
    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]  # Colors for the stacked bars
    p = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume",
               toolbar_location=None, tools="")
    p.vbar_stack(volume_labels, x='distance_bin', width=0.9, color=colors, source=source,
                 legend_label=volume_labels)
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"
    p.legend.title = "Log Volume"
    p.legend.label_text_font_size = "10pt"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Count", "@$name")]
    p.add_tools(hover)
    p.add_layout(p.legend[0], 'right')

    return p


def distance_Vs_occupide_histogram(df):
    df = df.copy()
    df = df[df['Count'] > 0]
    distance_bins = [0, 1000, 2000, 3000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4055"]
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))
    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]  # Colors for the stacked bars
    p = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume Occupied",
               toolbar_location=None, tools="")
    p.vbar_stack(volume_labels, x='distance_bin', width=0.9, color=colors, source=source,
                 legend_label=volume_labels)
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"
    p.legend.title = "Log Volume"
    p.legend.label_text_font_size = "10pt"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Count", "@$name")]
    p.add_tools(hover)
    p.add_layout(p.legend[0], 'right')

    return p


def distance_Vs_Volume_circle(df):
    df = df.copy()
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    p = figure(title='Distance to Center vs. Volume',
               output_backend="webgl", x_range=(0, 8110), y_range=(0, 8110))
    circle_center_x = 4055
    circle_center_y = 4055
    radius_values = [1000, 2000, 3000, 4055 * 1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000+']
    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5)
        if i < len(labels):  # Skip labeling for the largest circle
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)
    legend_items = []
    colors = Category20[20]
    scatter_renderers = []
    grouped = df.groupby(['lower bin', 'upper bin'])
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5)
        scatter_renderers.append(scatter)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))
    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'),
                                 ('Droplet', '@Droplet'),
                                 ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    return p

def distance_Vs_occupide_circle(df):
    df = df.copy()
    df = df[df['Count'] > 0]
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    p = figure(title='Distance to Center vs. Volume Occupied',
               output_backend="webgl", x_range=(0, 8110), y_range=(0, 8110))
    circle_center_x = 4055
    circle_center_y = 4055
    radius_values = [1000, 2000, 3000, 4055 * 1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000+']
    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5)
        if i < len(labels):  # Skip labeling for the largest circle
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)
    legend_items = []
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    colors = Category20[20]
    sources = []  # List to store all sources
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        sources.append(source)  # Add source to the list
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))
        scatter_renderers.append(scatter)
    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'),
                                 ('Droplet', '@Droplet'),
                                 ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)

    return p


def distance_Vs_Volume_colored_by_death_rate(df, data_dict,chip):
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    max_death_rate = []
    droplet_ids = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log)
            value['slope'] = value['log_count'].rolling(window_size).apply(
                lambda x: linregress(range(window_size), x)[0])
            if chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
                max_death_rate.append(value['slope'].max())
            else:
                max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
    death_rates = pd.DataFrame({'Slope': max_death_rate, 'Droplet': droplet_ids})
    death_rates.dropna(subset=['Slope'], inplace=True)  # Drop rows with NaN values
    df = pd.merge(df, death_rates, on='Droplet')
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-2,
                                     high=2)
    if chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
        title = 'Distance to Center vs. Volume Colored by Maximal Slope'
    else:
        title = 'Distance to Center vs. Volume Colored by Minimal Slope'
    p = figure(
        title=title,
        match_aspect=True,
        output_backend="webgl", width=800, height=600
    )
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"
    circle_center_x = 4055
    circle_center_y = 4055
    for radius in [4055*1.04, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5)
        label_text = f'{radius - 1000}-{radius}' if radius < 4055 else '3000+'
        label = Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05, text=label_text, text_align='center',
                      text_baseline='middle', text_font_style='bold', text_font_size='12pt')
        p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    sources = []
    checkbox_labels = []
    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)
        scatter = p.circle(x='X', y='Y',radius='radius', source=source, color={'field': 'Slope', 'transform': color_mapper}, fill_alpha=0.5)
        scatter_renderers.append(scatter)
        checkbox_labels.append(f'Bin {lower_bin}-{upper_bin}')
    checkbox_group = CheckboxGroup(labels=checkbox_labels, active=list(range(len(checkbox_labels))))
    checkbox_group.js_on_change('active',
                                CustomJS(args=dict(scatter_renderers=scatter_renderers, checkbox_group=checkbox_group),
                                         code="""
        for (let i = 0; i < scatter_renderers.length; i++) {
            scatter_renderers[i].visible = checkbox_group.active.includes(i);
        }
    """))
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Slope', '@{Slope}'), ('Droplet', '@Droplet')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Slope')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)

    layout_config = row(checkbox_group,p, sizing_mode='fixed', width=800, height=600)
    return layout_config

def distance_Vs_Volume_colored_by_fold_change(df, data_dict):
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    fold_change = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    fold_changes = pd.DataFrame({'Volume': Volume, 'fold change': fold_change, 'Droplet': droplet_id})
    df = pd.merge(df, fold_changes, on='Droplet')
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-10,
                                     high=10)
    p = figure(
        title='Distance to Center vs. Volume Colored by Fold Change',
        match_aspect=True,
        output_backend="webgl", width=800, height=600
    )
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"
    circle_center_x = 4055
    circle_center_y = 4055
    for radius in [4055*1.04, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5)
        label_text = f'{radius - 1000}-{radius}' if radius < 4055 else '3000+'
        label = Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05, text=label_text, text_align='center',
                      text_baseline='middle', text_font_style='bold', text_font_size='12pt')
        p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    checkbox_labels = []
    sources = []
    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)  # Add source to the list
        scatter = p.circle(x='X', y='Y',radius='radius', source=source, color={'field': 'fold change', 'transform': color_mapper}, fill_alpha=0.5)
        scatter_renderers.append(scatter)
        checkbox_labels.append(f'Bin {lower_bin}-{upper_bin}')
    checkbox_group = CheckboxGroup(labels=checkbox_labels, active=list(range(len(checkbox_labels))))
    checkbox_group.js_on_change('active',
                                CustomJS(args=dict(scatter_renderers=scatter_renderers, checkbox_group=checkbox_group),
                                         code="""
            for (let i = 0; i < scatter_renderers.length; i++) {
                scatter_renderers[i].visible = checkbox_group.active.includes(i);
            }
        """))
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Fold Change', '@{fold change}'), ('Droplet', '@Droplet')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='fold change')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)
    layout_config = row(checkbox_group,p, sizing_mode='fixed', width=800, height=600)
    return layout_config

def bins_volume_Vs_distance(data_dict,chip):
    volumes = []
    distances = []
    droplet_ids = []
    fold_changes = []
    death_rates=[]
    google_drive_urls = []
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            volumes.append(value['log_Volume'].iloc[0])
            distances.append(value['distance_to_center'].iloc[0])
            droplet_ids.append(key)
            google_drive_urls.append(value['Google Drive Link'].iloc[0])
            if value['Count'].iloc[-4:].mean() == 0:
                fold_changes.append(np.nan)
            else:
                fold_changes.append(value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log10)
            value['slope'] = value['log_count'].rolling(window_size).apply(lambda x: linregress(range(window_size), x)[0])
            if chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
                death_rates.append(value['slope'].max())
            else:
                death_rates.append(value['slope'].min())
    df = pd.DataFrame({'Volume': volumes, 'Distance': distances, 'Droplet': droplet_ids, 'Fold Change': fold_changes, 'Slope': death_rates, 'Google Drive Link': google_drive_urls})
    df['Fold Change'] = np.log2(df['Fold Change'])
    df['Fold Change'] = np.where(np.isnan(df['Fold Change']), min_fc, df['Fold Change'])
    df['upper_volume_bin'] = df['Volume'].apply(math.ceil)
    df['lower_volume_bin'] = df['Volume'].apply(math.floor)
    distance_bins = [0, 1000, 2000, 3000, float('inf')]
    df['lower_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[:-1], right=False)
    df['upper_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[1:], right=False)
    df['mean_fold_change'] = np.nan
    df['mean_slope'] = np.nan
    grouped=df.groupby(['lower_volume_bin', 'upper_volume_bin', 'lower_distance_bin', 'upper_distance_bin'], observed=True)
    for (lower_volume_bin, upper_volume_bin, lower_distance_bin, upper_distance_bin), group in grouped:
        mean_fold_change = group['Fold Change'].mean()
        mean_slope = group['Slope'].mean()
        df['Slope'] = df['Slope'].fillna(mean_slope)
        df.loc[(df['lower_volume_bin'] == lower_volume_bin) & (df['upper_volume_bin'] == upper_volume_bin) & (df['lower_distance_bin'] == lower_distance_bin) & (df['upper_distance_bin'] == upper_distance_bin), 'mean_fold_change'] = mean_fold_change
        df.loc[(df['lower_volume_bin'] == lower_volume_bin) & (df['upper_volume_bin'] == upper_volume_bin) & (df['lower_distance_bin'] == lower_distance_bin) & (df['upper_distance_bin'] == upper_distance_bin), 'mean_slope'] = mean_slope
    def create_plot(results, y_axis_label, y_column,points_column):
        if y_axis_label == 'Mean Fold Change':
            title = 'Distance Bin vs. Mean Fold Change'
        elif y_axis_label == 'Mean Slope' and chip=='C4- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
            title = 'Distance Bin vs. Maximal Slope'
        elif y_axis_label == 'Mean Slope':
            title = 'Distance Bin vs. Minimal Slope'
        p = figure(title=title, x_axis_label='Distance Bin', y_axis_label=y_axis_label,
                   output_backend="webgl", width=800, height=600)
        volume_bins = results['lower_volume_bin'].sort_values().unique()
        colors = Category20[len(volume_bins)]
        legend_items = []
        sources=[]
        scatter_renderers = []
        for i, volume_bin in enumerate(volume_bins):
            volume_bin_data = results[results['lower_volume_bin'] == volume_bin]
            volume_bin_data = volume_bin_data.sort_values('lower_distance_bin')
            source = ColumnDataSource(volume_bin_data)
            sources.append(source)
            line = p.line(x='lower_distance_bin', y=y_column, source=source, line_width=3, color=colors[i])
            scatter = p.scatter(x='lower_distance_bin', y=points_column,source=source, color=colors[i], fill_alpha=0.5)
            scatter_renderers.append(scatter)
            violin_renderers = []
            for lower_distance_bin in volume_bin_data['lower_distance_bin'].unique():
                data = volume_bin_data[volume_bin_data['lower_distance_bin'] == lower_distance_bin]
                if len(data) < 2:
                    continue
                kde = gaussian_kde(data[points_column]+np.random.normal(0, 1e-6, len(data[points_column])))
                y = np.linspace(data[points_column].min(), data[points_column].max(), 1000)
                density = kde(y)
                density=density/density.max()*250
                x_centered = lower_distance_bin
                x_combined = np.concatenate([x_centered + density, x_centered - density[::-1]])
                y_combined = np.concatenate([y, y[::-1]])
                violin=p.patch(x=x_combined, y=y_combined, fill_color=colors[i],line_color=colors[i], fill_alpha=0.3)
                violin_renderers.append(violin)
            legend_item = LegendItem(label=f'Volume Bin {volume_bin}-{volume_bin + 1}', renderers=[line,scatter]+violin_renderers)
            legend_items.append(legend_item)
        hover = HoverTool(tooltips=[
            ('Distance Bin', '@lower_distance_bin'),
            (points_column, f'@{{{points_column}}}'),
            ('Volume Bin', '@lower_volume_bin')
        ], renderers=scatter_renderers)
        p.add_tools(hover)
        legend = Legend(items=legend_items, location='top_left')
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'
        taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
            for (let source of sources) {
                const selected_index = source.selected.indices[0];
                if (selected_index != null) {
                    const data = source.data;
                    const url = data['Google Drive Link'][selected_index];
                    window.open(url, "_blank");
                    break;  // Open the first selected link and exit the loop
                }
            }
        """))
        p.add_tools(taptool)
        return p

    plot_fold_change = create_plot(df, 'Mean Fold Change', 'mean_fold_change','Fold Change')
    plot_death_rate = create_plot(df, 'Mean Slope', 'mean_slope','Slope')
    return row(plot_fold_change, plot_death_rate)


def dashborde():
    chips = split_data_to_chips()
    initial_densities = initial_stats(chips)
    for key, value in initial_densities.items():
        density=value['Count'].sum()/value['Volume'].sum()
        initial_densities[key]=density
    for key, value in chips.items():
        chips[key] = find_droplet_location(value)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]
    initial_data = initial_stats(chips)
    layouts={}
    for key, value in initial_data.items():
        chip, experiment_time, time_steps = get_slice(chips, key)
        # stats_box_plot=stats_box(value, experiment_time, time_steps, key)
        # droplets_histogram_plot=droplet_histogram(value)
        # Initial_Density_Vs_Volume_plot,volume=Initial_Density_Vs_Volume(value, initial_densities[key])
        # N0_Vs_Volume_plot=N0_Vs_Volume(value,volume)
        # Fraction_in_each_bin_plot=Fraction_in_each_bin(chip, experiment_time)
        # growth_curves_plot=growth_curves(chip)
        # normalize_growth_curves_plot=normalize_growth_curves(chip)
        # fold_change_plot=fold_change(chip,volume)
        # last_4_hours_average_plot=last_4_hours_average(chip,volume)
        # death_rate_by_droplets_plot=death_rate_by_droplets(chip,key)
        # death_rate_by_bins_plot=death_rate_by_bins(chip)
        # distance_Vs_Volume_histogram_plot=distance_Vs_Volume_histogram(value)
        # distance_Vs_occupide_histogram_plot=distance_Vs_occupide_histogram(value)
        # distance_Vs_Volume_circle_plot=distance_Vs_Volume_circle(value)
        # distance_Vs_occupide_circle_plot=distance_Vs_occupide_circle(value)
        distance_Vs_Volume_colored_by_death_rate_plot=distance_Vs_Volume_colored_by_death_rate(value, chip,key)
        distance_Vs_Volume_colored_by_fold_change_plot=distance_Vs_Volume_colored_by_fold_change(value, chip)
        bins_volume_Vs_distance_plot=bins_volume_Vs_distance(chip,key)
        layout = column(
                        # stats_box_plot,
                        # row(droplets_histogram_plot, N0_Vs_Volume_plot),
                        # row(Initial_Density_Vs_Volume_plot,Fraction_in_each_bin_plot),
                        # growth_curves_plot,
                        # normalize_growth_curves_plot,
                        # row(death_rate_by_droplets_plot, death_rate_by_bins_plot),
                        # row(fold_change_plot, last_4_hours_average_plot),
                        # row(distance_Vs_Volume_histogram_plot, distance_Vs_occupide_histogram_plot),
                        # row(distance_Vs_Volume_circle_plot, distance_Vs_occupide_circle_plot),
                        row(distance_Vs_Volume_colored_by_death_rate_plot, distance_Vs_Volume_colored_by_fold_change_plot,spacing=75),
                        bins_volume_Vs_distance_plot
                        )
        layouts[key]=layout
    return layouts


def create_dashboard():
    layouts = dashborde()
    output_file('dashboard.html')
    select = Select(title="Select Chip", options=list(layouts.keys()), value=list(layouts.keys())[0])
    all_layouts_column = column(*[layout for layout in layouts.values()], name="all_layouts")
    for layout in all_layouts_column.children:
        layout.visible = False
    layouts[select.value].visible = True
    select.js_on_change('value',
                        CustomJS(args=dict(layouts=layouts, all_layouts_column=all_layouts_column, select=select), code="""
        // Hide all layouts
        for (const layout of all_layouts_column.children) {
            layout.visible = false;
        }
        // Show the selected layout
        layouts[select.value].visible = true;
    """))
    show(column(select, all_layouts_column))

if __name__ == '__main__':
    create_dashboard()

