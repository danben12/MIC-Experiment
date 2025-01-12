import comparisons
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category20
from bokeh.models import HoverTool, TapTool
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.models import Range1d

plot_types, chip_names = comparisons.dashborde()

def overlay_histograms(plot_types, chip_names):
    histograms = [plot.children[0] for plot in plot_types['droplet_histogram']]
    hist = figure(title='Histogram of Droplet Size', x_axis_type='log',
                  x_axis_label='Volume', y_axis_label='Frequency', output_backend="webgl", width=1600, height=1200)
    legend_items = []
    colors = Category20[20]  # Use a palette with enough colors
    for i, h in enumerate(histograms):
        for j, renderer in enumerate(h.renderers):
            renderer.glyph.line_color = colors[i * 2 + j % 2]
            renderer.glyph.fill_color = colors[i * 2 + j % 2]
            renderer.visible = False
            hist.renderers.append(renderer)
            if j == 0:
                label = f'total droplets {chip_names[i]}'
                renderer.glyph.line_alpha = 0.5  # Set alpha for the first renderer
                renderer.glyph.fill_alpha = 0.3
            elif j == 1:
                label = f'occupied droplets {chip_names[i]}'
                renderer.glyph.line_alpha = 0.1  # Set alpha for the second renderer
                renderer.glyph.fill_alpha = 0.6
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
    legend = Legend(items=legend_items)
    hist.add_layout(legend, 'right')
    hist.legend.click_policy = 'hide'
    return hist

def N0_Vs_Volume_overlay(plot_types,chip_names):
    N0_Vs_Volume = [plot.children[0] for plot in plot_types['N0_Vs_Volume']]
    scatter = figure(title='N0 vs. Volume', x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='N0', output_backend="webgl", width=1600, height=1200)
    legend_items = []
    colors = Category20[20]  # Use a palette with enough colors
    for i, s in enumerate(N0_Vs_Volume):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            renderer.visible = False
            scatter.renderers.append(renderer)
            if j == 0:
                label = f'N0 vs. Volume {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            elif j == 1:
                label = f'Linear Regression {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                scatter.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    scatter.add_tools(hover)
    scatter.add_tools(TapTool())
    legend = Legend(items=legend_items)
    scatter.add_layout(legend, 'right')
    scatter.legend.click_policy = 'hide'
    return scatter
def Initial_Density_Vs_Volume_overlay(plot_types,chip_names):
    Initial_Density_Vs_Volume =plot_types['Initial_Density_Vs_Volume']
    scatter = figure(title='Initial Density vs. Volume', x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='Initial Density', output_backend="webgl", width=1600, height=1200)
    legend_items = []
    colors = Category20[20]  # Use a palette with enough colors
    for i, s in enumerate(Initial_Density_Vs_Volume):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            renderer.visible = False
            scatter.renderers.append(renderer)
            if j == 0:
                label = f'Initial density vs. Volume {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            elif j == 1:
                label = f'Rolling Mean {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            elif j == 2:
                label = f'Initial Density {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            elif j == 3:
                scatter.y_range.start = 2e-5
                scatter.y_range.end = 1e-1
                label = f'Vc {chip_names[i]}'
                span_location = renderer.location if hasattr(renderer, 'location') else 0
                source = ColumnDataSource(
                        data=dict(x=[span_location, span_location], y=[scatter.y_range.start, scatter.y_range.end]))
                renderer = scatter.line('x', 'y', source=source, line_color=colors[i * 2 + j % 2],
                                                 line_width=3, line_alpha=1,line_dash='dashed')
                renderer.visible = False
            else:
                continue
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                scatter.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    scatter.add_tools(hover)
    scatter.add_tools(TapTool())
    legend = Legend(items=legend_items)
    scatter.add_layout(legend, 'right')
    scatter.legend.click_policy = 'hide'
    return scatter
def Fraction_in_each_bin_overlay(plot_types,chip_names):
    Fraction_in_each_bin=plot_types['Fraction_in_each_bin']
    scatter = figure(title='Fraction of Population in Each Bin at Start and End of Simulation',
                     x_axis_label='Bin Range', y_axis_label='Fraction of Population', output_backend="webgl",
                     y_range=(0,100), width=1600, height=1200)
    legend_items = []
    colors = Category20[20]  # Use a palette with enough colors
    for i, s in enumerate(Fraction_in_each_bin):
        for j, renderer in enumerate(s.renderers):
            renderer.visible = False
            scatter.renderers.append(renderer)
            if j == 0:
                label = f'Start {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            elif j == 1:
                label = f'End {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
    legend = Legend(items=legend_items)
    scatter.add_layout(legend, 'right')
    scatter.legend.click_policy = 'hide'
    return scatter
def growth_curves_overlay(plot_types,chip_names):
    growth_curves_lin = [plot.children[0] for plot in plot_types['growth_curves']]
    growth_curves_log = [plot.children[1] for plot in plot_types['growth_curves']]
    fig_linear=figure(title='Growth Curves', x_axis_label='Time', y_axis_label='Mean Count',
               width=1600, height=1200, output_backend="webgl")
    fig_linear.y_range = Range1d(0, 1e7)
    fig_log=figure(title='Growth Log Scale', x_axis_label='Time', y_axis_label='Mean Count', width=1600, height=1200, output_backend="webgl", y_axis_type='log')
    fig_log.y_range = Range1d(1, 1e7)
    legend_items = []
    for i, g in enumerate(growth_curves_lin):
        for j in range(0, len(g.renderers), 2):
            renderer1 = g.renderers[j]
            renderer2 = g.renderers[j + 1] if j + 1 < len(g.renderers) else None
            renderer1.visible = False
            fig_linear.renderers.append(renderer1)
            if renderer2:
                renderer2.visible = False
                fig_linear.renderers.append(renderer2)
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1, renderer2]))
            else:
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1]))
    legend = Legend(items=legend_items)
    fig_linear.add_layout(legend, 'right')
    fig_linear.legend.click_policy = 'hide'
    legend_items = []
    for i, g in enumerate(growth_curves_log):
        for j in range(0, len(g.renderers), 2):
            renderer1 = g.renderers[j]
            renderer2 = g.renderers[j + 1] if j + 1 < len(g.renderers) else None
            renderer1.visible = False
            fig_log.renderers.append(renderer1)
            if renderer2:
                renderer2.visible = False
                fig_log.renderers.append(renderer2)
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1, renderer2]))
            else:
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1]))
    legend = Legend(items=legend_items)
    fig_log.add_layout(legend, 'right')
    fig_log.legend.click_policy = 'hide'
    return column(fig_linear,fig_log)
def normalize_growth_curves_overlay(plot_types,chip_names):
    normalize_growth_curves_lin = [plot.children[0] for plot in plot_types['normalize_growth_curves']]
    normalize_growth_curves_log = [plot.children[1] for plot in plot_types['normalize_growth_curves']]
    fig_linear = figure(title='Growth Curves', x_axis_label='Time', y_axis_label='Mean Count',
                        width=1600, height=1200, output_backend="webgl")
    fig_linear.y_range = Range1d(0, 1.3)
    fig_log = figure(title='Growth Log Scale', x_axis_label='Time', y_axis_label='Mean Count', width=1600, height=1200,
                     output_backend="webgl", y_axis_type='log')
    fig_log.y_range = Range1d(0.0085, 1.3)
    legend_items = []
    for i, g in enumerate(normalize_growth_curves_lin):
        for j in range(0, len(g.renderers), 2):
            renderer1 = g.renderers[j]
            renderer2 = g.renderers[j + 1] if j + 1 < len(g.renderers) else None
            renderer1.visible = False
            fig_linear.renderers.append(renderer1)
            if renderer2:
                renderer2.visible = False
                fig_linear.renderers.append(renderer2)
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1, renderer2]))
            else:
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1]))
    legend = Legend(items=legend_items)
    fig_linear.add_layout(legend, 'right')
    fig_linear.legend.click_policy = 'hide'
    legend_items = []
    for i, g in enumerate(normalize_growth_curves_log):
        for j in range(0, len(g.renderers), 2):
            renderer1 = g.renderers[j]
            renderer2 = g.renderers[j + 1] if j + 1 < len(g.renderers) else None
            renderer1.visible = False
            fig_log.renderers.append(renderer1)
            if renderer2:
                renderer2.visible = False
                fig_log.renderers.append(renderer2)
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1, renderer2]))
            else:
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1]))
    legend = Legend(items=legend_items)
    fig_log.add_layout(legend, 'right')
    fig_log.legend.click_policy = 'hide'
    return column(fig_linear, fig_log)




