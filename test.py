import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.palettes import Category10
import numpy as np

times = ['00h', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h', '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h', '24h']
results = pd.DataFrame(columns=['Chip', 'Time', 'Mean Bacteria Area', 'SE'])
for chip in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
    for time in times:
        df = pd.read_csv(rf'C:\Users\Owner\Desktop\MIC experiment\Results_files\Count\{chip}\{time}_{chip}_Simple Segmentation.csv')
        mean_bacteria_area = np.round(df['Total Area'].sum() / df['Count'].sum())
        se = np.round(df['Total Area'].std() / df['Count'].std())
        new_row = pd.DataFrame({'Chip': [chip], 'Time': [time], 'Mean Bacteria Area': [mean_bacteria_area], 'SE': [se], 'Mean Bacteria Area + SE': [mean_bacteria_area + se], 'Mean Bacteria Area - SE': [mean_bacteria_area - se]})
        results = pd.concat([results, new_row], ignore_index=True)

results['Time'] = pd.Categorical(results['Time'], categories=times, ordered=True)
p = figure(title="Mean Bacteria Area Over Time for Each Chip", x_axis_label='Time', y_axis_label='Mean Bacteria Area', x_range=times, width=800, height=600)
chips = results['Chip'].unique()
colors = Category10[len(chips)]
legend_items = []

for i, chip in enumerate(chips):
    chip_data = results[results['Chip'] == chip]
    source = ColumnDataSource(chip_data)
    line = p.line(x='Time', y='Mean Bacteria Area', source=source, line_width=2, color=colors[i])
    # varea = p.varea(x='Time', y1='Mean Bacteria Area + SE', y2='Mean Bacteria Area - SE', source=source, fill_alpha=0.2, color=colors[i])
    legend_item = LegendItem(label=chip, renderers=[line])
    legend_items.append(legend_item)

legend = Legend(items=legend_items)
p.add_layout(legend, 'right')
p.legend.click_policy = 'hide'

show(p)