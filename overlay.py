import comparisons
from bokeh.plotting import figure
from bokeh.io import show, output_file
from bokeh.models import Legend, LegendItem, GlyphRenderer, Span, CustomJS, Scatter, Quad, Segment, Line, \
    CheckboxGroup, Patch, Label, HoverTool, TapTool, ColumnDataSource, Range1d,Select
from bokeh.palettes import Category20
from bokeh.layouts import column, row


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
    return fig_linear,fig_log
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
    return fig_linear, fig_log

def fold_change_overlay(plot_types,chip_names):
    fold_change=plot_types['fold_change']
    fig=figure(title='Volume vs. Log2 Fold Change', x_axis_type='log',
                     x_axis_label='Volume', y_axis_label='Log2 Fold Change', output_backend="webgl",width=1600, height=1200)
    legend_items = []
    colors = Category20[20]
    for i, s in enumerate(fold_change):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            renderer.visible = False
            fig.renderers.append(renderer)
            if j == 0:
                label = f'Fold Change {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            elif j == 1:
                label = f'Moving Average {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            elif j == 2:
                label = f'Metapopulation Fold Change {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            elif j == 3:
                label = f'Vc {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
            else:
                continue
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    fig.y_range = Range1d(-10.5, 9)
    return fig
def last_4_hours_average_overlay(plot_types,chip_names):
    last_4_hours_average=plot_types['last_4_hours_average']
    fig=figure(title='Average Number of Bacteria in Last 4 Hours vs. Droplet Size', x_axis_type='log',
                     y_axis_type='log', x_axis_label='Volume', y_axis_label='Average Count', output_backend="webgl", width=1600, height=1200)
    legend_items = []
    colors = Category20[20]
    for i, s in enumerate(last_4_hours_average):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            renderer.visible = False
            fig.renderers.append(renderer)
            if j == 0:
                label = f'Average Count {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.fill_color = colors[i * 2 + j % 2]
            elif j == 1:
                label = f'Regression Before {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
                line_renderer = renderer  # Store the line renderer for the callback
            elif j == 2:
                label_annotation = renderer
                label_annotation.text_font_size = '12pt'
                label_annotation.text_font_style = 'bold'
                label_annotation.text_color = colors[i * 2 + j % 2]
                callback = CustomJS(args=dict(line=line_renderer, label=label_annotation), code="""
                        label.visible = line.visible;
                    """)
                line_renderer.js_on_change('visible', callback)
                continue
            elif (j == 3 or j==4) and isinstance(renderer, GlyphRenderer):
                source = renderer.data_source
                if 'x' in source.data and 'y' in source.data:
                    if all(x == 0 for x in source.data['x']) and all(y == 0 for y in source.data['y']):
                        continue
                label = f'Regression After {chip_names[i]}'
                renderer.glyph.line_color = colors[i * 2 + j % 2]
                renderer.glyph.line_width = 3
                renderer.glyph.line_alpha = 1
                line_renderer = renderer
            elif j == 3 or j==5 and isinstance(renderer, Span):
                fig.y_range.start = 0.05
                fig.y_range.end = 1e6
                label = f'Vc {chip_names[i]}'
                span_location = renderer.location if hasattr(renderer, 'location') else 0
                source = ColumnDataSource(
                    data=dict(x=[span_location, span_location], y=[fig.y_range.start, fig.y_range.end]))
                renderer = fig.line('x', 'y', source=source, line_color=colors[i * 2 + j % 2],
                                        line_width=3, line_alpha=1, line_dash='dashed')
                renderer.visible = False
            elif j == 4 and isinstance(renderer, Label):
                label_annotation = renderer
                label_annotation.text_font_size = '12pt'
                label_annotation.text_font_style = 'bold'
                label_annotation.text_color = colors[i * 2 + j % 2]
                callback = CustomJS(args=dict(line=line_renderer, label=label_annotation), code="""
                        label.visible = line.visible;
                    """)
                line_renderer.js_on_change('visible', callback)
                continue
            else:
                continue
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig
def death_rate_by_droplets_overlay(plot_types,chip_names):
    death_rate_by_droplets=plot_types['death_rate_by_droplets']
    fig=figure(title='Minimal/Maximal slope by Droplets', x_axis_label='log 10 Volume', y_axis_label='slope',output_backend="webgl",width=1600, height=1200)
    legend_items = []
    colors = Category20[20]
    for i, s in enumerate(death_rate_by_droplets):
        combined_tooltips = []
        add=False
        quad_renderers = []
        segments_renderers = []
        line_renderers = []
        for j, renderer in enumerate(s.renderers):
            if isinstance(renderer.glyph, Scatter):
                add=True
                renderer.visible = False
                scatter_renderer = renderer
                fig.renderers.append(scatter_renderer)
                if i==0 or i==4:
                    label = f'Maximal slope by Droplets {chip_names[i]}'
                else:
                    label = f'Minimal slope by Droplets {chip_names[i]}'
                scatter_renderer.glyph.line_color = colors[i * 2]
                scatter_renderer.glyph.fill_color = colors[i * 2]
                continue
            elif isinstance(renderer.glyph,Quad):
                renderer.visible = False
                fig.renderers.append(renderer)
                renderer.glyph.fill_alpha=0.3
                renderer.glyph.line_alpha=0.5
                renderer.glyph.fill_color = colors[i * 2]
                renderer.glyph.line_color = colors[i * 2]
                quad_renderers.append(renderer)
                continue
            elif isinstance(renderer.glyph,Segment):
                renderer.visible = False
                fig.renderers.append(renderer)
                renderer.glyph.line_color = colors[i * 2]
                segments_renderers.append(renderer)
                continue
            elif isinstance(renderer.glyph,Line) and add and not renderer.glyph.line_dash:
                renderer.visible = False
                fig.renderers.append(renderer)
                renderer.glyph.line_color = colors[i * 2]
                line_renderers.append(renderer)
                continue
            else:
                continue
        legend_items.append(LegendItem(label=label, renderers=[scatter_renderer,*quad_renderers,*segments_renderers,*line_renderers]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig

def death_rate_by_bins_overlay(plot_types,chip_names):
    death_rate_by_bins=plot_types['death_rate_by_bins']
    fig = figure(title='Death Rate by Bins', x_axis_label='Time', y_axis_label='Slope', width=1600, height=1200,
               output_backend="webgl")
    legend_items = []
    for i, g in enumerate(death_rate_by_bins):
        for j in range(0, len(g.renderers), 2):
            renderer1 = g.renderers[j]
            renderer2 = g.renderers[j + 1] if j + 1 < len(g.renderers) else None
            renderer1.visible = False
            fig.renderers.append(renderer1)
            if renderer2:
                renderer2.visible = False
                fig.renderers.append(renderer2)
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1, renderer2]))
            else:
                label = f"{g.legend.items[j // 2].label['value']} {chip_names[i]}"  # Keep the original label
                legend_items.append(LegendItem(label=label, renderers=[renderer1]))
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig

def distance_Vs_Volume_histogram_overlay(plot_types,chip_names):
    distance_Vs_Volume_histogram=plot_types['distance_Vs_Volume_histogram']
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000","4000-5000","5000-6000","6000+"]
    fig = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log 10 Volume",
               toolbar_location=None, tools="", width=1600, height=1200, output_backend="webgl")
    legend_items = []
    colors = Category20[20]
    for i, h in enumerate(distance_Vs_Volume_histogram):
        combined_tooltips = []
        for j, renderer in enumerate(h.renderers):
            renderer.visible = False
            fig.renderers.append(renderer)
            if j == 0:
                label = f'3-4 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 1:
                label = f'4-5 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 2:
                label = f'5-6 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 3:
                label = f'6-7 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 4:
                label = f'7-8 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in h.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig

def distance_Vs_occupide_histogram_overlay(plot_types,chip_names):
    distance_Vs_occupide_histogram=plot_types['distance_Vs_occupide_histogram']
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000","4000-5000","5000-6000","6000+"]
    fig = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log 10 Volume Occupied",
               toolbar_location=None, tools="", width=1600, height=1200, output_backend="webgl")
    legend_items = []
    colors = Category20[20]
    for i, h in enumerate(distance_Vs_occupide_histogram):
        combined_tooltips = []
        for j, renderer in enumerate(h.renderers):
            renderer.visible = False
            fig.renderers.append(renderer)
            if j == 0:
                label = f'3-4 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 1:
                label = f'4-5 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 2:
                label = f'5-6 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 3:
                label = f'6-7 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            elif j == 4:
                label = f'7-8 {chip_names[i]}'
                renderer.glyph.fill_color = colors[j]
                renderer.glyph.line_color = colors[j]
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in h.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig
def distance_Vs_Volume_circle_overlay(plot_types,chip_names):
    distance_Vs_Volume_circle=plot_types['distance_Vs_Volume_circle']
    fig = figure(title='Distance to Center vs. Volume',
               output_backend="webgl", x_range=(0, 13000), y_range=(0, 13000), width=1600, height=1200)
    legend_items = []
    colors = Category20[20]
    for i, s in enumerate(distance_Vs_Volume_circle):
        combine_tooltips = []
        for j, renderer in enumerate(s.renderers):
            if i==0 and j in [0,1,2,3,4,5,6]:
                fig.renderers.append(renderer)
                continue
            elif j==7:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 3-4 {chip_names[i]}'
                renderer.glyph.fill_color = colors[0]
                renderer.glyph.line_color = colors[0]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 8:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 4-5 {chip_names[i]}'
                renderer.glyph.fill_color = colors[1]
                renderer.glyph.line_color = colors[1]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 9:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 5-6 {chip_names[i]}'
                renderer.glyph.fill_color = colors[2]
                renderer.glyph.line_color = colors[2]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 10:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 6-7 {chip_names[i]}'
                renderer.glyph.fill_color = colors[3]
                renderer.glyph.line_color = colors[3]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 11:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 7-8 {chip_names[i]}'
                renderer.glyph.fill_color = colors[4]
                renderer.glyph.line_color = colors[4]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            else:
                continue
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combine_tooltips.extend(tool.tooltips)
    hover = HoverTool(tooltips=combine_tooltips)
    fig.add_tools(hover)
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig

def distance_Vs_occupide_circle_overlay(plot_types,chip_names):
    distance_Vs_occupide_circle=plot_types['distance_Vs_occupide_circle']
    fig = figure(title='Distance to Center vs. Volume Occupied',
               output_backend="webgl", x_range=(0, 13000), y_range=(0, 13000), width=1600, height=1200)
    legend_items = []
    colors = Category20[20]
    for i, s in enumerate(distance_Vs_occupide_circle):
        combine_tooltips = []
        for j, renderer in enumerate(s.renderers):
            if i==0 and j in [0,1,2,3,4,5,6]:
                fig.renderers.append(renderer)
                continue
            elif j==7:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 3-4 {chip_names[i]}'
                renderer.glyph.fill_color = colors[0]
                renderer.glyph.line_color = colors[0]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 8:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 4-5 {chip_names[i]}'
                renderer.glyph.fill_color = colors[1]
                renderer.glyph.line_color = colors[1]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 9:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 5-6 {chip_names[i]}'
                renderer.glyph.fill_color = colors[2]
                renderer.glyph.line_color = colors[2]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 10:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 6-7 {chip_names[i]}'
                renderer.glyph.fill_color = colors[3]
                renderer.glyph.line_color = colors[3]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            elif j == 11:
                renderer.visible = False
                fig.renderers.append(renderer)
                label = f'Bin 7-8 {chip_names[i]}'
                renderer.glyph.fill_color = colors[4]
                renderer.glyph.line_color = colors[4]
                renderer.glyph.fill_alpha = 0.3
                renderer.glyph.line_alpha = 0.5
            else:
                continue
            legend_items.append(LegendItem(label=label, renderers=[renderer]))
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combine_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combine_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig.add_layout(legend, 'right')
    fig.legend.click_policy = 'hide'
    return fig
def distance_Vs_Volume_colored_by_death_rate_overlay(plot_types,chip_names):
    checkboxes = [plot.children[0] for plot in plot_types['distance_Vs_Volume_colored_by_death_rate']]
    plot=[plot.children[1] for plot in plot_types['distance_Vs_Volume_colored_by_death_rate']]
    fig = figure(
        title='Distance to Center vs. Volume Colored by Maximal/Minimal Slope',
        match_aspect=True,
        output_backend="webgl", width=1600, height=1200)
    combined_labels = []
    for i, s in enumerate(checkboxes):
        combined_labels.extend([f"{label} {chip_names[i]}" for label in s.labels])
    combined_checkbox_group = CheckboxGroup(labels=combined_labels)
    for i, s in enumerate(plot):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            if i == 0 and j in [0, 1, 2, 3, 4, 5, 6]:
                fig.renderers.append(renderer)
                continue
            elif i==1 and j==len(s.renderers)-1:
                fig.add_layout(renderer, 'right')
            elif j>=7 and j<len(s.renderers)-1:
                renderer.visible = False
                fig.renderers.append(renderer)
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover=HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    combined_checkbox_group.js_on_change('active', CustomJS(args=dict(renderers=fig.renderers[7:]), code="""
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = cb_obj.active.includes(i);
        }
    """))
    return row(combined_checkbox_group, fig)
def distance_Vs_Volume_colored_by_fold_change_overlay(plot_types,chip_names):
    checkboxes = [plot.children[0] for plot in plot_types['distance_Vs_Volume_colored_by_fold_change']]
    plot = [plot.children[1] for plot in plot_types['distance_Vs_Volume_colored_by_fold_change']]
    fig = figure(
        title='Distance to Center vs. Volume Colored by Fold Change',
        match_aspect=True,
        output_backend="webgl", width=1600, height=1200)
    combined_labels = []
    for i, s in enumerate(checkboxes):
        combined_labels.extend([f"{label} {chip_names[i]}" for label in s.labels])
    combined_checkbox_group = CheckboxGroup(labels=combined_labels)
    for i, s in enumerate(plot):
        combined_tooltips = []
        for j, renderer in enumerate(s.renderers):
            if i == 0 and j in [0, 1, 2, 3, 4, 5, 6]:
                fig.renderers.append(renderer)
                continue
            elif i == 1 and j == len(s.renderers) - 1:
                fig.add_layout(renderer, 'right')
            elif j >= 7 and j < len(s.renderers) - 1:
                renderer.visible = False
                fig.renderers.append(renderer)
                renderer.glyph.line_alpha = 0.5
                renderer.glyph.fill_alpha = 0.3
        for tool in s.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    combined_checkbox_group.js_on_change('active', CustomJS(args=dict(renderers=fig.renderers[7:]), code="""
            for (let i = 0; i < renderers.length; i++) {
                renderers[i].visible = cb_obj.active.includes(i);
            }
        """))
    return row(combined_checkbox_group, fig)
def bins_volume_Vs_distance_overlay(plot_types,chip_names):
    plot_fold_change=[plot.children[0] for plot in plot_types['bins_volume_Vs_distance']]
    plot_death_rate=[plot.children[1] for plot in plot_types['bins_volume_Vs_distance']]
    fig_fold_change = figure(title='Distance Bin vs. Mean Fold Change', x_axis_label='Distance Bin', y_axis_label='Mean Fold Change',
                   output_backend="webgl", width=1600, height=1200)
    fig_death_rate =figure(title='Distance Bin vs. Minimal/Maximal Slope', x_axis_label='Distance Bin', y_axis_label='Mean Slope',
                   output_backend="webgl", width=1600, height=1200)
    legend_items = []
    for i, g in enumerate(plot_fold_change):
        line_counter = 0
        combined_tooltips = []
        for j, renderer in enumerate(g.renderers):
            legend_item_renderers = []
            # Start with a Line glyph
            if isinstance(renderer.glyph, Line):
                renderer.visible = False
                legend_item_renderers.append(renderer)
                fig_fold_change.renderers.append(renderer)
                j += 1
                # Add the Scatter glyph
                if j < len(g.renderers) and isinstance(g.renderers[j].glyph, Scatter):
                    g.renderers[j].visible = False
                    legend_item_renderers.append(g.renderers[j])
                    fig_fold_change.renderers.append(g.renderers[j])
                    j += 1
                    # Add all Patch glyphs
                    while j < len(g.renderers) and isinstance(g.renderers[j].glyph, Patch):
                        g.renderers[j].visible = False
                        legend_item_renderers.append(g.renderers[j])
                        fig_fold_change.renderers.append(g.renderers[j])
                        j += 1
                label = f"Volume bin {3 + line_counter}-{4 + line_counter} {chip_names[i]}"
                legend_items.append(LegendItem(label=label, renderers=legend_item_renderers))
                line_counter += 1
        for tool in g.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig_fold_change.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig_fold_change.add_tools(hover)
    fig_fold_change.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig_fold_change.add_layout(legend, 'right')
    fig_fold_change.legend.click_policy = 'hide'
    legend_items = []
    for i, g in enumerate(plot_death_rate):
        line_counter = 0
        combined_tooltips = []
        for j, renderer in enumerate(g.renderers):
            legend_item_renderers = []
            # Start with a Line glyph
            if isinstance(renderer.glyph, Line):
                renderer.visible = False
                legend_item_renderers.append(renderer)
                fig_death_rate.renderers.append(renderer)
                j += 1
                # Add the Scatter glyph
                if j < len(g.renderers) and isinstance(g.renderers[j].glyph, Scatter):
                    g.renderers[j].visible = False
                    legend_item_renderers.append(g.renderers[j])
                    fig_death_rate.renderers.append(g.renderers[j])
                    j += 1
                    # Add all Patch glyphs
                    while j < len(g.renderers) and isinstance(g.renderers[j].glyph, Patch):
                        g.renderers[j].visible = False
                        legend_item_renderers.append(g.renderers[j])
                        fig_death_rate.renderers.append(g.renderers[j])
                        j += 1
                label = f"Volume bin {3 + line_counter}-{4 + line_counter} {chip_names[i]}"
                legend_items.append(LegendItem(label=label, renderers=legend_item_renderers))
                line_counter += 1
        for tool in g.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig_death_rate.js_on_event('tap', tool.callback)
    hover = HoverTool(tooltips=combined_tooltips)
    fig_death_rate.add_tools(hover)
    fig_death_rate.add_tools(TapTool())
    legend = Legend(items=legend_items)
    fig_death_rate.add_layout(legend, 'right')
    fig_death_rate.legend.click_policy = 'hide'
    return fig_fold_change,fig_death_rate


def FC_vs_density_overlay(plot_types, chip_names):
    FC_vs_density = plot_types['FC_vs_density']
    fig = figure(title='Log2 Fold Change vs. Log2 Density colored by Volume', x_axis_label='Log2 Density',
                 y_axis_label='Log2 Fold Change', output_backend="webgl", width=1600, height=1200)
    renderers = []
    labels = []
    for i, g in enumerate(FC_vs_density):
        combined_tooltips = []
        fig.renderers.extend(g.renderers[:-1])
        renderers.extend(g.renderers[:-1])
        labels.extend([f"{chip_names[i]}"])
        for renderer in g.renderers:
            renderer.visible = False
        if i == 0:
            colorbar = g.renderers[-1]
            colorbar.visible = True
            fig.add_layout(colorbar, 'right')
        for tool in g.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)

    checkbox_group = CheckboxGroup(labels=labels, active=[])
    checkbox_group.js_on_change('active', CustomJS(args=dict(renderers=renderers), code="""
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = cb_obj.active.includes(i);
        }
    """))

    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    layout = row(checkbox_group, fig)
    return layout

def FC_vs_volume_overlay(plot_types, chip_names):
    FC_vs_volume = plot_types['FC_vs_Volume']
    fig = figure(title='Log2 Fold Change vs. Log2 Volume colored by Density', x_axis_label='Log2 Volume', y_axis_label='Log2 Fold Change',output_backend="webgl", width=1600, height=1200)
    renderers = []
    labels = []
    for i, g in enumerate(FC_vs_volume):
        combined_tooltips = []
        if i == 0:
            fig.add_layout(g.renderers[-1], 'right')
        fig.renderers.extend(g.renderers[:-1])
        renderers.extend(g.renderers[:-1])
        labels.extend([f"{chip_names[i]}"])
        for renderer in g.renderers:
            renderer.visible = False
        if i == 0:
            colorbar = g.renderers[-1]
            colorbar.visible = True
            fig.add_layout(colorbar, 'right')
        for tool in g.tools:
            if isinstance(tool, HoverTool):
                combined_tooltips.extend(tool.tooltips)
            elif isinstance(tool, TapTool):
                fig.js_on_event('tap', tool.callback)

    checkbox_group = CheckboxGroup(labels=labels, active=[])
    checkbox_group.js_on_change('active', CustomJS(args=dict(renderers=renderers), code="""
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = cb_obj.active.includes(i);
        }
    """))
    hover = HoverTool(tooltips=combined_tooltips)
    fig.add_tools(hover)
    fig.add_tools(TapTool())
    layout = row(checkbox_group, fig)
    return layout





def overlay_dashboard(plot_types,chip_names):
    lin_growth_curves, log_growth_curves = growth_curves_overlay(plot_types, chip_names)
    lin_norm_growth_curves, log_norm_growth_curves = normalize_growth_curves_overlay(plot_types, chip_names)
    distance_bin_fold_change, distance_bin_death_rate = bins_volume_Vs_distance_overlay(plot_types, chip_names)
    figures = {
        "Histogram": overlay_histograms(plot_types, chip_names),
        "N0 vs Volume": N0_Vs_Volume_overlay(plot_types, chip_names),
        "Initial Density vs Volume": Initial_Density_Vs_Volume_overlay(plot_types, chip_names),
        "Fraction in Each Bin": Fraction_in_each_bin_overlay(plot_types, chip_names),
        "Linear Scale Growth Curves": lin_growth_curves,
        "Log Scale Growth Curves": log_growth_curves,
        "Linear Scale Normalized Growth Curves": lin_norm_growth_curves,
        "Log Scale Normalized Growth Curves": log_norm_growth_curves,
        "Fold Change": fold_change_overlay(plot_types, chip_names),
        "Last 4 Hours Average": last_4_hours_average_overlay(plot_types, chip_names),
        "Death Rate by Droplets": death_rate_by_droplets_overlay(plot_types, chip_names),
        "Death Rate by Bins": death_rate_by_bins_overlay(plot_types, chip_names),
        "Distance vs Volume Histogram": distance_Vs_Volume_histogram_overlay(plot_types, chip_names),
        "Distance vs Occupied Histogram": distance_Vs_occupide_histogram_overlay(plot_types, chip_names),
        "Distance vs Volume Circle": distance_Vs_Volume_circle_overlay(plot_types, chip_names),
        "Distance vs Occupied Circle": distance_Vs_occupide_circle_overlay(plot_types, chip_names),
        "Distance vs Volume Colored by Death Rate": distance_Vs_Volume_colored_by_death_rate_overlay(plot_types,
                                                                                                     chip_names),
        "Distance vs Volume Colored by Fold Change": distance_Vs_Volume_colored_by_fold_change_overlay(plot_types,
                                                                                                       chip_names),
        "Distance Bin vs. Mean Fold Change": distance_bin_fold_change,
        "Distance Bin vs. Minimal/Maximal Slope": distance_bin_death_rate,
        "FC vs Density": FC_vs_density_overlay(plot_types, chip_names),
        "FC vs Volume": FC_vs_volume_overlay(plot_types, chip_names)
    }
    return figures


def create_dashboard(plot_types, chip_names):
    figures = overlay_dashboard(plot_types, chip_names)
    output_file("overlay.html")
    select = Select(title="Select Plot", options=list(figures.keys()), value=list(figures.keys())[0])
    layout = column(select, *figures.values())

    for fig in figures.values():
        fig.visible = False
    figures[select.value].visible = True

    select.js_on_change('value', CustomJS(args=dict(figures=figures), code="""
        for (let key in figures) {
            figures[key].visible = false;
        }
        figures[cb_obj.value].visible = true;
    """))

    show(layout)
if __name__ == '__main__':
    plot_types, chip_names = comparisons.dashborde()
    create_dashboard(plot_types, chip_names)





