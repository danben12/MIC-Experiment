from bokeh.layouts import column, row
from bokeh.models import Select, CustomJS, Spacer, Range1d
from bokeh.plotting import output_file, show

import main
from main import normalize_growth_curves, N0_Vs_Volume, fold_change, last_4_hours_average, death_rate_by_droplets


def dashborde():
    chips = main.split_data_to_chips()
    initial_densities = main.initial_stats(chips)
    for key, value in initial_densities.items():
        density=value['Count'].sum()/value['Volume'].sum()
        initial_densities[key]=density
    for key, value in chips.items():
        chips[key] = main.find_droplet_location(value)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]
    initial_data = main.initial_stats(chips)
    plot_types = {
        # 'droplet_histogram': [],
        # 'N0_Vs_Volume': [],
        # 'Initial_Density_Vs_Volume': [],
        # 'Fraction_in_each_bin': [],
        # 'growth_curves': [],
        # 'normalize_growth_curves': [],
        # 'fold_change': [],
        # 'last_4_hours_average': [],
        # 'death_rate_by_droplets': [],
        # 'death_rate_by_bins': [],
        # 'distance_Vs_Volume_histogram': [],
        # 'distance_Vs_occupide_histogram': [],
        # 'distance_Vs_Volume_circle': [],
        # 'distance_Vs_occupide_circle': [],
        'distance_Vs_Volume_colored_by_death_rate': [],
        'distance_Vs_Volume_colored_by_fold_change': [],
        # 'bins_volume_Vs_distance':[]
    }
    for key, value in initial_data.items():
        chip, experiment_time, time_steps = main.get_slice(chips, key)
        # droplet_histogram_column = main.droplet_histogram(value)
        # droplet_histogram_column.children[0].update(title=f'Histogram of Droplet Size for {key}')
        # droplet_histogram_column.children[0].y_range = Range1d(start=0, end=350)
        # plot_types['droplet_histogram'].append(droplet_histogram_column)
        Initial_Density_Vs_Volume_column,volume = main.Initial_Density_Vs_Volume(value, initial_densities[key])
        # Initial_Density_Vs_Volume_column.update(title=f'Initial Density Vs Volume for {key}')
        # Initial_Density_Vs_Volume_column.y_range = Range1d(start=10**(-4.7), end=10**(-0.5))
        # plot_types['Initial_Density_Vs_Volume'].append(Initial_Density_Vs_Volume_column)
        # N0_Vs_Volume_column = N0_Vs_Volume(value,volume)
        # N0_Vs_Volume_column.children[0].update(title=f'N0 Vs Volume for {key}')
        # N0_Vs_Volume_column.children[0].y_range = Range1d(start=1, end=10**5)
        # plot_types['N0_Vs_Volume'].append(N0_Vs_Volume_column)
        # Fraction_in_each_bin_column = main.Fraction_in_each_bin(chip, experiment_time)
        # Fraction_in_each_bin_column.update(title=f'Fraction of Population in Each Bin at Start for {key}')
        # plot_types['Fraction_in_each_bin'].append(Fraction_in_each_bin_column)
        # growth_curves_column = main.growth_curves(chip)
        # growth_curves_column = column(
        #     growth_curves_column.children[0],  # Linear scale plot
        #     growth_curves_column.children[1]  # Log scale plot
        # )
        # growth_curves_column.children[0].update(title=f'Growth Curves for {key}')
        # growth_curves_column.children[1].update(title=f'Growth Curves log scale for {key}')
        # growth_curves_column.children[0].y_range = Range1d(start=0, end=10**7)
        # growth_curves_column.children[1].y_range = Range1d(start=1, end=10**7)
        # plot_types['growth_curves'].append(growth_curves_column)
        # normalize_growth_curves_column = normalize_growth_curves(chip)
        # normalize_growth_curves_column = column(
        #     normalize_growth_curves_column.children[0],  # Linear scale plot
        #     normalize_growth_curves_column.children[1]  # Log scale plot
        # )
        # normalize_growth_curves_column.children[0].update(title=f'Normalized Growth Curves for {key}')
        # normalize_growth_curves_column.children[1].update(title=f'Normalized Growth Curves log scale for {key}')
        # normalize_growth_curves_column.children[0].y_range = Range1d(start=0, end=1.3)
        # normalize_growth_curves_column.children[1].y_range = Range1d(start=0.0085, end=1.3)
        # plot_types['normalize_growth_curves'].append(normalize_growth_curves_column)
        # fold_change_column = fold_change(chip, volume)
        # fold_change_column.update(title=f'Fold Change for {key}')
        # fold_change_column.y_range = Range1d(start=-10.5, end=9)
        # plot_types['fold_change'].append(fold_change_column)
        # last_4_hours_average_column = last_4_hours_average(chip,volume)
        # last_4_hours_average_column.update(title=f'Average Number of Bacteria in Last 4 Hours for {key}')
        # last_4_hours_average_column.y_range = Range1d(start=0.05, end=10**6)
        # plot_types['last_4_hours_average'].append(last_4_hours_average_column)
        # death_rate_by_droplets_column = death_rate_by_droplets(chip,key)
        # death_rate_by_droplets_column.update(title=f'Slope by Droplet Size for {key}')
        # death_rate_by_droplets_column.y_range = Range1d(start=-1.2, end=1.8)
        # plot_types['death_rate_by_droplets'].append(death_rate_by_droplets_column)
        # death_rate_by_bins_column = main.death_rate_by_bins(chip)
        # death_rate_by_bins_column.y_range = Range1d(start=-2.5, end=1.2)
        # death_rate_by_bins_column.update(title=f'Slope by Bin for {key}')
        # plot_types['death_rate_by_bins'].append(death_rate_by_bins_column)
        # distance_Vs_Volume_histogram_column = main.distance_Vs_Volume_histogram(value)
        # distance_Vs_Volume_histogram_column.update(title=f'Normalized Stacked Histogram: Distance vs. Log Volume for {key}')
        # plot_types['distance_Vs_Volume_histogram'].append(distance_Vs_Volume_histogram_column)
        # distance_Vs_occupide_histogram_column = main.distance_Vs_occupide_histogram(value)
        # distance_Vs_occupide_histogram_column.update(title=f'Normalized Stacked Histogram: Distance vs. Log Volume Occupied for {key}')
        # plot_types['distance_Vs_occupide_histogram'].append(distance_Vs_occupide_histogram_column)
        # distance_Vs_Volume_circle_column = main.distance_Vs_Volume_circle(value)
        # distance_Vs_Volume_circle_column.update(title=f'Distance to Center vs. Volume for {key}')
        # plot_types['distance_Vs_Volume_circle'].append(distance_Vs_Volume_circle_column)
        # distance_Vs_occupide_circle_column = main.distance_Vs_occupide_circle(value)
        # distance_Vs_occupide_circle_column.update(title=f'Distance to Center vs. Volume Occupied for {key}')
        # plot_types['distance_Vs_occupide_circle'].append(distance_Vs_occupide_circle_column)
        distance_Vs_Volume_colored_by_death_rate_column = main.distance_Vs_Volume_colored_by_death_rate(value, chip, key)
        distance_Vs_Volume_colored_by_death_rate_column.children[1].update(title=f'Distance to Center vs. Volume Colored by Slope for {key}')
        plot_types['distance_Vs_Volume_colored_by_death_rate'].append(distance_Vs_Volume_colored_by_death_rate_column)
        distance_Vs_Volume_colored_by_fold_change_column = main.distance_Vs_Volume_colored_by_fold_change(value, chip)
        distance_Vs_Volume_colored_by_fold_change_column.children[1].update(title=f'Distance to Center vs. Volume Colored by Fold Change for {key}')
        plot_types['distance_Vs_Volume_colored_by_fold_change'].append(distance_Vs_Volume_colored_by_fold_change_column)
        # bins_volume_Vs_distance_column = main.bins_volume_Vs_distance(chip, key)
        # bins_volume_Vs_distance_column = column(
        #     bins_volume_Vs_distance_column.children[0],
        #     bins_volume_Vs_distance_column.children[1]
        # )
        # bins_volume_Vs_distance_column.children[0].y_range = Range1d(start=-10.5, end=10)
        # bins_volume_Vs_distance_column.children[1].y_range = Range1d(start=-1.2, end=0.8)
        # bins_volume_Vs_distance_column.children[0].update(title=f'Bin volumes vs. Distance to Center for {key} by Mean Fold Change')
        # bins_volume_Vs_distance_column.children[1].update(title=f'Bin volumes vs. Distance to Center for {key} by Mean Death Rate')
        # plot_types['bins_volume_Vs_distance'].append(bins_volume_Vs_distance_column)

    return plot_types,list(chips.keys())


def create_dashboard():
    plot_types, chips_names = dashborde()
    output_file('comparisons.html')

    select = Select(title="Select plot", options=list(plot_types.keys()), value=list(plot_types.keys())[0])
    first_chip_selection = Select(title="Select chip", options=chips_names, value=chips_names[0])
    second_chip_selection = Select(title="Select chip", options=chips_names, value=chips_names[1])

    # Create a layout for the selected plot and chips with a spacer in between
    spacer = Spacer(width=75)
    plot_layout = row(plot_types[select.value][0], spacer, plot_types[select.value][1])

    # Create a CustomJS callback to update the plot layout
    callback = CustomJS(args=dict(plot_types=plot_types, chips_names=chips_names,
                                  select=select, first_chip_selection=first_chip_selection,
                                  second_chip_selection=second_chip_selection, plot_layout=plot_layout, spacer=spacer),
                        code="""
        var selected_plot = select.value;
        var first_chip = first_chip_selection.value;
        var second_chip = second_chip_selection.value;
        var first_chip_index = chips_names.indexOf(first_chip);
        var second_chip_index = chips_names.indexOf(second_chip);
        plot_layout.children = [plot_types[selected_plot][first_chip_index],
                                    spacer,
                                    plot_types[selected_plot][second_chip_index]];
    """)

    # Create a CustomJS callback to update the options of the second chip selection
    update_second_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection,
                                                    second_chip_selection=second_chip_selection,
                                                    chips_names=chips_names),
                                          code="""
        var first_chip = first_chip_selection.value;
        var options = chips_names.filter(chip => chip !== first_chip);
        second_chip_selection.options = options;
        if (second_chip_selection.value === first_chip) {
            second_chip_selection.value = options[0];
        }
    """)

    # Create a CustomJS callback to update the options of the first chip selection
    update_first_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection,
                                                   second_chip_selection=second_chip_selection,
                                                   chips_names=chips_names),
                                         code="""
        var second_chip = second_chip_selection.value;
        var options = chips_names.filter(chip => chip !== second_chip);
        first_chip_selection.options = options;
        if (first_chip_selection.value === second_chip) {
            first_chip_selection.value = options[0];
        }
    """)

    # Add the CustomJS callback to the select widgets
    select.js_on_change('value', callback)
    first_chip_selection.js_on_change('value', callback)
    second_chip_selection.js_on_change('value', callback)

    # Add the CustomJS callback to update the chip options
    first_chip_selection.js_on_change('value', update_second_chip_options)
    second_chip_selection.js_on_change('value', update_first_chip_options)

    layout = column(select, first_chip_selection, second_chip_selection, plot_layout)
    show(layout)

if __name__ == '__main__':
    create_dashboard()
