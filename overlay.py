from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import Select, CustomJS
from bokeh.plotting import figure
import comparisons
from bokeh.models import LinearScale, LogScale

def get_axis_type(fig1):
    axis_types = []
    if isinstance(fig1.x_scale, LinearScale):
        axis_types.append('linear')
    else:
        axis_types.append('log')
    if isinstance(fig1.y_scale, LinearScale):
        axis_types.append('linear')
    else:
        axis_types.append('log')
    return axis_types


def combine_legends(fig1, fig2, first_chip, second_chip):
    fig1_legend = fig1.legend[0]
    fig2_legend = fig2.legend[0]
    for item in fig1_legend.items:
        item.label.value += f' ({first_chip})'
    for item in fig2_legend.items:
        item.label.value += f' ({second_chip})'
    fig1_legend.items.extend(fig2_legend.items)
    return fig1_legend

def create_dashboard():
    plot_types, chips_names = comparisons.dashborde()
    output_file('comparisons.html')
    select: Select = Select(title="Select plot", options=list(plot_types.keys()), value=list(plot_types.keys())[0])
    first_chip_selection = Select(title="Select chip", options=chips_names, value=chips_names[0])
    second_chip_selection = Select(title="Select chip", options=chips_names, value=chips_names[1])
    axis_types = get_axis_type(plot_types[select.value][0])
    combined = figure(title=f'Combined plot {select.value}', x_axis_type=axis_types[0], y_axis_type=axis_types[1], width=1200, height=900)
    combined_legend = combine_legends(plot_types[select.value][0], plot_types[select.value][1],first_chip_selection.value,second_chip_selection.value)
    combined.add_layout(combined_legend, 'right')
    default_plots_to_combine = [plot_types[select.value][0], plot_types[select.value][1]]
    combined_renderers = []
    for plot in default_plots_to_combine:
        combined_renderers.extend(plot.renderers)
    combined.renderers = combined_renderers
    overlay_callback = CustomJS(args=dict(plot_types=plot_types, chips_names=chips_names,
                                  select=select, first_chip_selection=first_chip_selection,
                                  second_chip_selection=second_chip_selection, combined=combined),
                        code="""
        var selected_plot = select.value;
        var first_chip = first_chip_selection.value;
        var second_chip = second_chip_selection.value;
        var first_chip_index = chips_names.indexOf(first_chip);
        var second_chip_index = chips_names.indexOf(second_chip);
        var first_plot = plot_types[selected_plot][first_chip_index];
        var second_plot = plot_types[selected_plot][second_chip_index];
        combined.renderers = first_plot.renderers.concat(second_plot.renderers);
    """)
    update_second_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection, second_chip_selection=second_chip_selection, chips_names=chips_names), code="""
        var first_chip = first_chip_selection.value;
        var options = chips_names.filter(function(chip) { return chip != first_chip; });
        second_chip_selection.options = options;
        if (second_chip_selection.value == first_chip) {
            second_chip_selection.value = options[0];
        }
    """)
    update_first_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection, second_chip_selection=second_chip_selection, chips_names=chips_names), code="""
        var second_chip = second_chip_selection.value;
        var options = chips_names.filter(function(chip) { return chip != second_chip; });
        first_chip_selection.options = options;
        if (first_chip_selection.value == second_chip) {
            first_chip_selection.value = options[0];
        }
    """)
    select.js_on_change('value', overlay_callback)
    first_chip_selection.js_on_change('value', overlay_callback)
    second_chip_selection.js_on_change('value', overlay_callback)
    first_chip_selection.js_on_change('value', update_second_chip_options)
    second_chip_selection.js_on_change('value', update_first_chip_options)
    layout = column(select, first_chip_selection, second_chip_selection, combined)
    show(layout)

if __name__ == '__main__':
    create_dashboard()