import pandas as pd
import numpy as np
from scipy.stats import linregress
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Legend, LegendItem, HoverTool, Select, CustomJS, ColumnDataSource
from bokeh.palettes import Turbo256
from bokeh.io import curdoc, output_file

# Read the data from the Excel file
df = pd.read_csv('cell to pixels.csv')

# Drop rows with NaN values
df = df.dropna(subset=['pixel count', 'cell count'])

# Drop rows where both 'pixel count' and 'cell count' are 0
df = df[(df['pixel count'] != 0) & (df['cell count'] != 0)]

# Group the data by 'slice'
grouped = df.groupby('Slice')

# Create a scatter plot for each slice
plots = {}
for slice_value, group in grouped:
    p = figure(title=f'Scatter Plot for Slice: {slice_value}',
               x_axis_label='Pixel Count', y_axis_label='Cell Count',
               width=1200, height=800)  # Increase figure size

    legend_items = []
    unique_times = group['time'].unique()
    colors = [Turbo256[i] for i in range(0, 256, 256 // 25)]

    for i, (time, time_group) in enumerate(group.groupby('time')):
        color = colors[i % len(colors)]

        # Scatter plot
        source = ColumnDataSource(data=dict(x=time_group['pixel count'], y=time_group['cell count']))
        scatter = p.scatter('x', 'y', size=10, color=color, source=source)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(time_group['pixel count'], time_group['cell count'])

        # Plot the regression line
        x = np.linspace(time_group['pixel count'].min(), time_group['pixel count'].max(), 100)
        y = slope * x + intercept
        line_source = ColumnDataSource(data=dict(x=x, y=y))
        line = p.line('x', 'y', color=color, source=line_source)

        legend_label = f'Time: {time}, Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value ** 2:.2f}'
        legend_items.append(LegendItem(label=legend_label, renderers=[scatter, line]))

    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Pixel Count", "@x"),
        ("Cell Count", "@y"),
    ])
    p.add_tools(hover)
    legend = Legend(items=legend_items)
    p.add_layout(legend, 'right')
    p.legend.click_policy = 'hide'
    for renderer in p.renderers:
        renderer.visible = False
    plots[str(slice_value)] = p

def create_dash(plots):
    output_file('cell to pixels.html')
    select = Select(title='Select Slice:', options=list(plots.keys()), value=list(plots.keys())[0])
    layout = column(select, *plots.values())
    for fig in plots.values():
        fig.visible = False
    plots[select.value].visible = True
    select.js_on_change('value', CustomJS(args=dict(figures=plots), code="""
        for (let key in figures) {
            figures[key].visible = false;
        }
        figures[cb_obj.value].visible = true;
    """))
    show(layout)
if __name__ == '__main__':
    create_dash(plots)