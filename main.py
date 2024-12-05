import math
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from bokeh.io import output_file
from bokeh.models import Div, HoverTool,TapTool, ColorBar, LogTicker, LinearColorMapper, BasicTicker, Legend, LegendItem,BoxAnnotation, Span, Whisker,Label,Spacer
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category20,RGB
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CDSView,Select,CustomJS
from numba.cpython.unsafe.numbers import viewer
from scipy.stats import linregress, alpha
from bokeh.transform import linear_cmap
import plotly.graph_objects as go
from matplotlib import cm





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
    bins = np.logspace(np.log10(df['Volume'].min()), np.log10(df['Volume'].max()), num=16)
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


def N0_Vs_Volume(df):
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
    filtered_df = filtered_df[filtered_df['Volume'] > 1000]
    filtered_df = filtered_df[filtered_df['Volume'] > np.mean(filtered_df['Volume'])]
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


def Initial_Density_Vs_Volume(df):
    df['initial density'] = df['Count'] / df['Volume']
    source = ColumnDataSource(df)
    view = CDSView()
    scatter = figure(title='Initial Density vs. Volume', x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='Initial Density', output_backend="webgl")
    scatter_renderer = scatter.scatter('Volume', 'initial density', source=source, view=view, color='gray', alpha=1,
                                       legend_label='Initial Density vs. Volume')
    hover = HoverTool(
        tooltips=[('Volume', '@Volume'), ('Initial Density', '@{initial density}'), ('Droplet ID', '@Droplet')],
        renderers=[scatter_renderer])
    scatter.add_tools(hover)

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)

    filtered_sorted_df = df[df['initial density'] > 0].sort_values(by='Volume')
    rolling_mean = np.log10(filtered_sorted_df['initial density']).rolling(window=50).mean()
    scatter.line(filtered_sorted_df['Volume'], 10 ** rolling_mean, color='red', legend_label='Rolling Mean')
    return scatter

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
    p = figure(title='Fraction of Population in Each Bin at Start and End of Simulation', x_axis_label='Bin Range', y_axis_label='Fraction of Population')
    bin_centers = [(start_bin + end_bin) / 2 for start_bin, end_bin in bins.index]
    bin_width = 0.4
    p.vbar(x=[bin_center - bin_width / 2 for bin_center in bin_centers], top=bins['start fraction'], width=bin_width, color='blue', legend_label='Start')
    p.vbar(x=[bin_center + bin_width / 2 for bin_center in bin_centers], top=bins['end fraction'], width=bin_width, color='green', legend_label='End')
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p

def fold_change(data_dict):
    fold_change = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])  # Array to store Google Drive URLs
    min_fc = 1
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
                if value['Count'].iloc[-4:].mean() / value['Count'].iloc[0] < min_fc:
                    min_fc = value['Count'].iloc[-4:].mean() / value['Count'].iloc[0]
    min_fc = min_fc * 0.9
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    fold_change = np.log2(fold_change)
    df = pd.DataFrame({'Volume': Volume, 'fold change': fold_change, 'Droplet': droplet_id, 'Google Drive Link': google_drive_url})
    df = df.sort_values(by='Volume').reset_index(drop=True)
    sub_df = df[df['fold change'] > np.log2(min_fc)].reset_index(drop=True)
    sub_df['moving average'] = sub_df['fold change'].rolling(window=50).mean()
    source = ColumnDataSource(df)
    sub_source = ColumnDataSource(sub_df)
    view = CDSView()
    sub_view = CDSView()
    scatter = figure(title='Volume vs. Log2 Fold Change', x_axis_type='log', y_axis_type='linear',
                     x_axis_label='Volume', y_axis_label='Log2 Fold Change', output_backend="webgl")
    scatter.scatter('Volume', 'fold change', source=source, view=view, color='gray', alpha=1,
                    legend_label='Volume vs. Log2 Fold Change')
    scatter.line('Volume', 'moving average', source=sub_source, view=sub_view, color='red', legend_label='Moving Average')
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

def last_4_hours_average(chip):
    last_4_hours = {droplet_id: df[df['time'] > 20].reset_index(drop=True) for droplet_id, df in chip.items()}
    average_counts = np.array([df['Count'].mean() for df in last_4_hours.values()])
    min_average_count = min(value for value in average_counts if value > 0) * 0.9
    average_counts = np.where(average_counts == 0, min_average_count, average_counts)
    droplet_sizes = [df['Volume'].iloc[0] for df in chip.values()]
    droplet_ids = [df['Droplet'].iloc[0] for df in chip.values()]
    google_drive_urls = [df['Google Drive Link'].iloc[0] for df in chip.values()]  # Add URL
    data = pd.DataFrame({'Volume': droplet_sizes, 'Average Count': average_counts, 'Droplet': droplet_ids, 'Google Drive Link': google_drive_urls})
    data = data.sort_values(by='Volume').reset_index(drop=True)
    sub_data = data[data['Average Count'] > data['Average Count'].min()].reset_index(drop=True)
    sub_data['moving average'] = sub_data['Average Count'].rolling(window=50).mean().reset_index(drop=True)
    source = ColumnDataSource(data)
    view = CDSView()
    sub_source = ColumnDataSource(sub_data)
    sub_view = CDSView()
    scatter = figure(title='Average Number of Bacteria in Last 4 Hours vs. Droplet Size', x_axis_type='log',
                     y_axis_type='log', x_axis_label='Volume', y_axis_label='Average Count', output_backend="webgl")
    scatter.scatter('Volume', 'Average Count', source=source, view=view, color='gray', alpha=1,
                    legend_label='Average Count')
    scatter.line('Volume', 'moving average', source=sub_source, view=sub_view, color='red', legend_label='moving average')
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Average Count', '@{Average Count}'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter.renderers[0]])
    scatter.add_tools(hover)

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
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

def death_rate_by_droplets(data_dict):
    volumes = []
    max_death_rate = []
    droplet_ids = []
    google_drive_urls = []  # Array to store Google Drive URLs
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
            max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
            google_drive_urls.append(value['Google Drive Link'].iloc[0])  # Add URL

    df = pd.DataFrame({'Volume': np.log10(volumes), 'Max Death Rate': max_death_rate, 'Droplet': droplet_ids, 'Google Drive Link': google_drive_urls})
    df['upper bin'] = df['Volume'].apply(math.ceil)
    df['lower bin'] = df['Volume'].apply(math.floor)
    # Calculate the metapopulation over time
    valid_droplets = [value for key, value in data_dict.items() if value['Count'].iloc[0] != 0]
    metapopulation_df = pd.concat(valid_droplets, ignore_index=True)
    metapopulation = metapopulation_df.groupby('time')['Count'].sum().reset_index(name='metapopulation')

    # Perform a rolling linear regression with a window size of 4
    window_size = 4
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0]
    )

    # Find the most negative slope
    most_negative_slope = metapopulation['slope'].min()

    # Use the most negative slope as the death rate of the metapopulation
    mean_death_rate = most_negative_slope

    p = figure(title='Death Rate by Droplets', x_axis_label='log 10 Volume', y_axis_label='Max Death Rate',
               output_backend="webgl")
    grouped = df.groupby(['lower bin', 'upper bin'])
    colors = Category20[20]
    scatter_renderers = []
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index % len(colors)]  # Cycle through the palette if more bins than colors
        source = ColumnDataSource(group)
        view = CDSView()
        q1 = group['Max Death Rate'].quantile(0.25)
        q2 = group['Max Death Rate'].quantile(0.5)  # Median
        q3 = group['Max Death Rate'].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = group['Max Death Rate'][group['Max Death Rate'] <= upper_whisker].max()
        lower_whisker = group['Max Death Rate'][group['Max Death Rate'] >= lower_whisker].min()
        p.quad(top=[q3], bottom=[q1], left=[lower_bin], right=[upper_bin], fill_color=color, alpha=0.3)
        p.segment(x0=[lower_bin], y0=[q2], x1=[upper_bin], y1=[q2], line_color="black")
        p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[upper_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q3],
                  line_color="black")
        p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[lower_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q1],
                  line_color="black")
        p.line(x=[lower_bin, upper_bin], y=[upper_whisker, upper_whisker], line_color="black")
        p.line(x=[lower_bin, upper_bin], y=[lower_whisker, lower_whisker], line_color="black")
        scatter = p.scatter(x='Volume', y='Max Death Rate', source=source,view=view, color=color)
        scatter_renderers.append(scatter)
        taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
            }
        """), renderers=[scatter])
        p.add_tools(taptool)


    # Add dashed line for mean death rate
    p.line(x=[3, 8], y=[mean_death_rate, mean_death_rate], line_dash='dashed', line_color='black',
           line_width=2, legend_label='Meta-population Death Rate')

    p.tools = [tool for tool in p.tools if not isinstance(tool, HoverTool)]
    hover = HoverTool(
        tooltips=[('Index', '$index'),('Log 10 Volume', '@Volume'), ('Max Death Rate', '@{Max Death Rate}'), ('Droplet', '@Droplet')],
        renderers=scatter_renderers)
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
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['lower bin'] = df['log_Volume'].apply(math.floor)

    p = figure(title='Distance to Center vs. Volume',
               output_backend="webgl", x_range=(0, 8110), y_range=(0, 8110))

    circle_center_x = 4055  # Assuming the center is at (4055, 4055)
    circle_center_y = 4055
    circle_radius = 4055 * 1.04  # Assuming the radius is 4055

    p.circle(x=[circle_center_x], y=[circle_center_y], radius=circle_radius, line_color="black", fill_color=None,
             alpha=0.5)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=3000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 3000, text='3000+', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=2000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 2000, text='2000-3000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=1000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 1000, text='1000-2000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    label = Label(x=circle_center_x, y=circle_center_y, text='0-1000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)

    grouped = df.groupby(['lower bin', 'upper bin'])
    colors = Category20[20]
    scatter_renderers = []
    legend_items = []

    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index % len(colors)]
        source = ColumnDataSource(group)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5)
        scatter_renderers.append(scatter)
        legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter])
        legend_items.append(legend_item)

    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Droplet', '@Droplet'), ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    p.add_tools(taptool)

    return p

def distance_Vs_occupide_circle(df):
    df = df.copy()
    df = df[df['Count'] > 0]
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    p = figure(title='Distance to Center vs. Volume Occupied',
               output_backend="webgl", x_range=(0, 8110), y_range=(0, 8110))
    circle_center_x = 4055  # Assuming the center is at (4055, 4055)
    circle_center_y = 4055
    circle_radius = 4055 * 1.04  # Assuming the radius is 4055
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=circle_radius, line_color="black", fill_color=None,
             alpha=0.5)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=3000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 3000, text='3000+', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=2000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 2000, text='2000-3000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    p.circle(x=[circle_center_x], y=[circle_center_y], radius=1000, line_color="black", fill_color=None, alpha=0.5)
    label = Label(x=circle_center_x, y=circle_center_y + 1000, text='1000-2000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    label = Label(x=circle_center_x, y=circle_center_y, text='0-1000', text_align='center',
                  text_baseline='middle', text_font_style='bold', text_font_size='12pt')
    p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    colors = Category20[20]
    scatter_renderers = []
    legend_items = []

    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index % len(colors)]
        source = ColumnDataSource(group)
        view = CDSView()
        scatter = p.scatter(x='X', y='Y', source=source, view=view, color=color)
        scatter_renderers.append(scatter)
        legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter])
        legend_items.append(legend_item)

    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    p.tools = [tool for tool in p.tools if not isinstance(tool, HoverTool)]
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Droplet', '@Droplet')], renderers=scatter_renderers)
    p.add_tools(hover)

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    p.add_tools(taptool)

    return p


def distance_Vs_Volume_colored_by_death_rate(df, data_dict):
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
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
            max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
    death_rates = pd.DataFrame({'Max Death Rate': max_death_rate, 'Droplet': droplet_ids})
    death_rates.dropna(subset=['Max Death Rate'], inplace=True)  # Drop rows with NaN values
    df = pd.merge(df, death_rates, on='Droplet')
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=df['Max Death Rate'].min(),
                                     high=df['Max Death Rate'].max())
    p = figure(
        title='Distance to Center vs. Volume Colored by Death Rate',
        match_aspect=True,
        output_backend="webgl", width=800, height=600
    )
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"
    circle_center_x = 4055
    circle_center_y = 4055
    for radius in [4055, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5)
        label_text = f'{radius - 1000}-{radius}' if radius != 4055 else '3000+'
        label = Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05, text=label_text, text_align='center',
                      text_baseline='middle', text_font_style='bold', text_font_size='12pt')
        p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    legend_items = []
    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        scatter = p.scatter(x='X', y='Y', source=source, color={'field': 'Max Death Rate', 'transform': color_mapper})
        scatter_renderers.append(scatter)
        legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter])
        legend_items.append(legend_item)
    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Death Rate', '@{Max Death Rate}'), ('Droplet', '@Droplet')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Death Rate')
    p.add_layout(color_bar, 'right')

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    p.add_tools(taptool)

    layout_config = row(p, sizing_mode='fixed', width=800, height=600)
    return layout_config


# def deathrate_volume_by_distance(data_dict):
#     distances = []
#     max_death_rate = []
#     volumes=[]
#     for key, value in data_dict.items():
#         if value['Count'].iloc[0] == 0:
#             continue
#         else:
#             window_size = 4
#             mask = value['Count'] > 0
#             value['log_count'] = value[mask]['Count'].apply(np.log)
#             value['slope'] = value['log_count'].rolling(window_size).apply(lambda x: linregress(range(window_size), x)[0])
#             distances.append(value['distance_to_center'].iloc[0])
#             volumes.append(value['Volume'].iloc[0])
#             max_death_rate.append(value['slope'].min())
#     df = pd.DataFrame({'volumes':np.log10(volumes),'distances': distances, 'Max Death Rate': max_death_rate})
#     data_min=df['distances'].min()
#     data_max=df['distances'].max()
#     bin_edges = np.arange(data_min, data_max + 1000, 1000)
#     df['bin_index'] = np.digitize(df['distances'], bins=bin_edges, right=False)
#     df['lower distance bin'] = bin_edges[df['bin_index'] - 1]
#     df['upper distance bin'] = bin_edges[df['bin_index']]
#     df['lower volume bin'] = df['volumes'].apply(math.floor)
#     df['upper volume bin'] = df['volumes'].apply(math.ceil)
#     grouped = df.groupby(['lower distance bin', 'upper distance bin', 'lower volume bin', 'upper volume bin'])['Max Death Rate'].agg(['mean', 'std']).reset_index()
#     grouped['std'] = grouped['std'] / 2
#     grouped['distance_bin_middle'] = (grouped['lower distance bin'] + grouped['upper distance bin']) / 2
#     p = figure(title='Mean Max Death Rate by Distance', x_axis_label='Distance', y_axis_label='Mean Max Death Rate', output_backend="webgl",width=800, height=600)
#     volume_bins = grouped[['lower volume bin', 'upper volume bin']].drop_duplicates()
#     volume_bins = volume_bins.sort_values(by=['lower volume bin', 'upper volume bin'])
#     colors = Category20[20]
#     color_index = 0
#     legend_items = []
#     for _, row in volume_bins.iterrows():
#         lower_bin, upper_bin = row['lower volume bin'], row['upper volume bin']
#         filtered_data = grouped[
#             (grouped['lower volume bin'] == lower_bin) & (grouped['upper volume bin'] == upper_bin)].copy()
#         filtered_data['mean - std'] = filtered_data['mean'] - filtered_data['std']
#         filtered_data['mean + std'] = filtered_data['mean'] + filtered_data['std']
#         source = ColumnDataSource(filtered_data)
#         line = p.line(x='distance_bin_middle', y='mean', source=source, line_width=3, color=colors[color_index])
#         varea = p.varea(x='distance_bin_middle', y1='mean - std', y2='mean + std', source=source, fill_alpha=0.2,
#                         color=colors[color_index])
#         legend_item = LegendItem(label=f'Volume Bin {lower_bin}-{upper_bin}', renderers=[line, varea])
#         legend_items.append(legend_item)
#         color_index = (color_index + 1) % len(colors)
#     legend = Legend(items=legend_items, location="center")
#     p.add_layout(legend, 'right')
#     p.legend.click_policy = "hide"
#     return p
#
#
#
#
# def deathrate_by_distance(data_dict):
#     distances = []
#     max_death_rate = []
#     volumes = []
#     Droplet = []
#
#     for key, value in data_dict.items():
#         if value['Count'].iloc[0] == 0:
#             continue
#         window_size = 4
#         mask = value['Count'] > 0
#         value['log_count'] = value[mask]['Count'].apply(np.log)
#         value['slope'] = value['log_count'].rolling(window_size).apply(
#             lambda x: linregress(range(window_size), x)[0])
#         distances.append(value['distance_to_center'].iloc[0])
#         volumes.append(value['Volume'].iloc[0])
#         Droplet.append(value['Droplet'].iloc[0])
#         max_death_rate.append(value['slope'].min())
#
#     df = pd.DataFrame(
#         {'Droplet': Droplet, 'volumes': np.log10(volumes), 'distances': distances, 'Max Death Rate': max_death_rate})
#     data_min = 0
#     data_max = df['distances'].max()
#     bin_edges = np.arange(data_min, data_max + 1000, 1000)
#     df['bin_index'] = np.digitize(df['distances'], bins=bin_edges, right=False)
#     df['lower distance bin'] = bin_edges[df['bin_index'] - 1]
#     df['upper distance bin'] = bin_edges[df['bin_index']]
#     df['lower volume bin'] = df['volumes'].apply(np.floor)
#     df['upper volume bin'] = df['volumes'].apply(np.ceil)
#
#     p = figure(title='Death Rate by Distance', x_axis_label='distance', y_axis_label='Max Death Rate',
#                output_backend="webgl")
#     grouped = df.groupby(['lower distance bin', 'upper distance bin'])
#     colors = Category20[20]
#     scatter_renderers = []
#     for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
#         color = colors[index % len(colors)]
#         source = ColumnDataSource(group)
#         view = CDSView()
#         q1 = group['Max Death Rate'].quantile(0.25)
#         q2 = group['Max Death Rate'].quantile(0.5)
#         q3 = group['Max Death Rate'].quantile(0.75)
#         iqr = q3 - q1
#         upper_whisker = q3 + 1.5 * iqr
#         lower_whisker = q1 - 1.5 * iqr
#         upper_whisker = group['Max Death Rate'][group['Max Death Rate'] <= upper_whisker].max()
#         lower_whisker = group['Max Death Rate'][group['Max Death Rate'] >= lower_whisker].min()
#         p.quad(top=[q3], bottom=[q1], left=[lower_bin], right=[upper_bin], fill_color=color, alpha=0.3)
#         p.segment(x0=[lower_bin], y0=[q2], x1=[upper_bin], y1=[q2], line_color="black")
#         p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[upper_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q3],
#                   line_color="black")
#         p.segment(x0=[(lower_bin + upper_bin) / 2], y0=[lower_whisker], x1=[(lower_bin + upper_bin) / 2], y1=[q1],
#                   line_color="black")
#         p.line(x=[lower_bin, upper_bin], y=[upper_whisker, upper_whisker], line_color="black")
#         p.line(x=[lower_bin, upper_bin], y=[lower_whisker, lower_whisker], line_color="black")
#         scatter = p.scatter(x='distances', y='Max Death Rate', source=source, view=view, color=color)
#         scatter_renderers.append(scatter)
#     p.tools = [tool for tool in p.tools if not isinstance(tool, HoverTool)]
#     hover = HoverTool(
#         tooltips=[('Distance', '@distances'), ('Max Death Rate', '@{Max Death Rate}'), ('Droplet', '@Droplet')],
#         renderers=scatter_renderers)
#     p.add_tools(hover)
#     return p
#
# def foldchange_in_each_bin_per_distance(chip):
#     df = pd.concat(chip, ignore_index=True)
#     df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
#     df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
#     df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
#     data_max = df['distance_to_center'].max()
#     bin_edges = np.arange(0, data_max + 1000, 1000)
#     df['bin_index'] = np.digitize(df['distance_to_center'], bins=bin_edges, right=False)
#     df['lower distance bin'] = bin_edges[df['bin_index'] - 1]
#     df['upper distance bin'] = bin_edges[df['bin_index']]
#     grouped = df.groupby(['lower bin', 'upper bin', 'lower distance bin', 'upper distance bin', 'time'])['Count'].sum().reset_index(name='Count')
#     results=pd.DataFrame(columns=['lower bin', 'upper bin', 'lower distance bin', 'upper distance bin','fold change'])
#     for (lower_bin, upper_bin,lower_dis_bin,upper_dis_bin), group in grouped.groupby(['lower bin', 'upper bin', 'lower distance bin', 'upper distance bin']):
#         if group['Count'].iloc[0]==0:
#             continue
#         else:
#             fold_change =np.log2( group['Count'].iloc[-4:].mean() / group['Count'].iloc[0])
#             new_row = pd.DataFrame(
#                 {'lower bin': [lower_bin], 'upper bin': [upper_bin], 'lower distance bin': [lower_dis_bin],
#                  'upper distance bin': [upper_dis_bin], 'fold change': [fold_change]})
#             results = pd.concat([results, new_row], ignore_index=True)
#     p = figure(title='Fold Change in Each Bin per Distance', x_axis_label='Distance', y_axis_label='Log 2 Fold Change', output_backend="webgl",width=800, height=600)
#     grouped = results.groupby(['lower bin', 'upper bin'])
#     colors = Category20[20]
#     legend_items = []  # Create a list to store all LegendItem objects
#     for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
#         color = colors[index % len(colors)]
#         source = ColumnDataSource(group)
#         view = CDSView()
#         line = p.line(x='lower distance bin', y='fold change', source=source, line_width=3, color=color)
#         legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line])
#         legend_items.append(legend_item)  # Add each LegendItem to the list
#     legend = Legend(items=legend_items)
#     p.add_layout(legend, 'right')  # Place the legend outside the plot
#     p.legend.click_policy = 'hide'
#     return p

def dashborde():
    chips = split_data_to_chips()
    for key, value in chips.items():
        chips[key] = find_droplet_location(value)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]
    initial_data = initial_stats(chips)
    layouts={}
    for key, value in initial_data.items():
        chip, experiment_time, time_steps = get_slice(chips, key)
        layout = column(
                        #stats_box(value, experiment_time, time_steps, key),
                        #row(droplet_histogram(value), N0_Vs_Volume(value)),
                        #row(Initial_Density_Vs_Volume(value), Fraction_in_each_bin(chip, experiment_time)),
                        #growth_curves(chip),normalize_growth_curves(chip),
                        #row(fold_change(chip), last_4_hours_average(chip)),
                        row(death_rate_by_droplets(chip), death_rate_by_bins(chip)),
                        #row(distance_Vs_Volume_histogram(value),distance_Vs_occupide_histogram(value)),
                        #row(distance_Vs_Volume_circle(value),distance_Vs_occupide_circle(value)),
                        #row(distance_Vs_Volume_colored_by_death_rate(value, chip)),
                        #row(deathrate_by_distance(chip),deathrate_volume_by_distance(chip)),
                        #row(foldchange_in_each_bin_per_distance(chip))
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

