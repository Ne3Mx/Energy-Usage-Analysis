
#Import python libraries
import dask
from dask.distributed import Client
from dask import delayed
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import textwrap  # Import the textwrap module for wrapping text
from scipy.stats import norm, skewnorm
import pandas as pd
import scipy

# Function to remove illegal characters from a string when generating a filename
def clean_filename(filename):

    # Define a translation table mapping illegal characters to underscores
    illegal_chars = r'\/:"*?<>|'
    translation_table = str.maketrans(illegal_chars, '_' * len(illegal_chars))
    
    # Use translate() to replace illegal characters with underscores
    cleaned_filename = filename.translate(translation_table)

    return cleaned_filename

# Function to calculate and plot percentiles
def plot_percentiles(ax, percentiles, shape, loc, scale, color='grey'):
    for percentile in percentiles:
        value = skewnorm.ppf(percentile / 100.0, shape, loc, scale)
        ax.axvline(value, color=color, linestyle=':')
        ax.text(value,  ax.get_ylim()[1] * 0.05, f'{percentile}th Percentile', rotation=90, va='bottom', ha='right', fontsize=8, color='grey')

# Display the histogram for the specific sector
def compute_histogram(df_filtered, summary_stats_dict, loop_count, meter_df, meter_stats, baseline_stats, meter_count,sector, stack_height):
    
    if DISPLAY_RAW_DATA:
        #Display the filtered DataFrame
        print("\n\nEntering Compute\n\n")
        print(df_filtered)
    
    # Convert the 'P1' column to numeric, coercing non-numeric values to NaN
    df_filtered['P1Norm'] = pd.to_numeric(df_filtered['P1Norm'], errors='coerce')

    #If extreme outliers should be removed
    if REMOVE_EXTREME_OUTLIERS:
        # Get summary statistics of the filtered DataFrame
        summary_stats = df_filtered.describe()
        
        if DISPLAY_RAW_DATA:
            print("Summary stats", summary_stats)

        # Calculate Interquartile Range (IQR) bounds
        q1 = summary_stats.loc['25%']
        q3 = summary_stats.loc['75%']
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if DISPLAY_RAW_DATA:
            print("lower_bound\n", lower_bound)
            print("upper_bound\n", upper_bound)

        # Reset the index of the filtered DataFrame
        df_filtered = df_filtered.reset_index(drop=True)

        # Filter outliers based on the IQR bounds
        df_filtered = df_filtered[(df_filtered['P1Norm'] >= lower_bound['P1Norm']) & (df_filtered['P1Norm'] <= upper_bound['P1Norm'])]

    # Group the data by 'SNUMBER' and compute the mean and median of 'P1'
    filtered_data = df_filtered.groupby('SNUMBER')['P1Norm'].agg(['mean', 'median'])
    
    # Remove entries where mean == 0 or median == 0
    filtered_data = filtered_data[(filtered_data['mean'] != 0) & (filtered_data['median'] != 0)]

    # Count SNUMBERs where mean == 0 or median == 0
    count_zero = filtered_data[(filtered_data['mean'] == 0) | (filtered_data['median'] == 0)].shape[0]

    # Count SNUMBERs where mean != 0 or median != 0
    count_non_zero = filtered_data[(filtered_data['mean'] != 0) | (filtered_data['median'] != 0)].shape[0]

    if DISPLAY_RAW_DATA:
        print(f'\n\nCount of SNUMBERs where mean or median is 0: {count_zero}')
        print(f'Count of SNUMBERs where mean or median is not 0: {count_non_zero}\n\n')

    #Only plot if there is data
    if count_non_zero != 0:

        #Re-init stats
        
        # Alternative method to filter outliers based on Z-score

        # Calculate Z-scores for mean and median
        # grouped_data['mean_zscore'] = np.abs((grouped_data['mean'] - grouped_data['mean'].mean()) / grouped_data['mean'].std())
        # grouped_data['median_zscore'] = np.abs((grouped_data['median'] - grouped_data['median'].mean()) / grouped_data['median'].std())

        # Uncomment and use this block if needed
        # if REMOVE_EXTREME_OUTLIERS:
        #     # Define a threshold for outliers (e.g., Z-score > 3)
        #     threshold = 3
        #
        #     # Filter out outliers based on Z-score
        #     filtered_data = grouped_data[(grouped_data['mean_zscore'] <= threshold) & (grouped_data['median_zscore'] <= threshold)]
        # else:
        #     filtered_data = grouped_data

        # For now, using grouped_data directly
        # filtered_data = grouped_data

        # Set figure size and DPI
        plt.figure(dpi=150, figsize=(14, 6))

        # Plot histogram for the mean
        ax1 = plt.subplot(1, 2, 1)
        ax1.hist(filtered_data['mean'], bins=50, density=True, alpha=0.2, color='red', edgecolor='red')
        ax1.set_title(f'P1 Mean histogram from {meter_count} sampled meters of sector:\n{sector}\nTotal area of bars in histogram equals 1 (Density=True)\nConstructed with {stack_height:,} samples')
        ax1.set_xlabel('Mean (Wh/m^2, P1Norm)')
        ax1.set_ylabel('Density')

        try:
            # Fit a skewed normal distribution to the mean data
            shape, loc, scale = skewnorm.fit(filtered_data['mean'])

            # Generate points on the x-axis
            x = np.linspace(min(filtered_data['mean']), max(filtered_data['mean']), 1000)

            # Calculate the skewed normal distribution values
            pdf = skewnorm.pdf(x, shape, loc, scale)

            # Plot the skewed distribution over the histogram
            ax1.plot(x, pdf, color='black', linestyle='-', label='Skewed Normal Distribution')

            # Plot test Comparison mean if specified
            if PLOT_COMPARISON_METER:
                # Fill the area between the vertical lines and under the pdf
                x_fill = np.linspace(baseline_stats['mean'], meter_stats['mean'], 1000)
                y_fill = skewnorm.pdf(x_fill, shape, loc, scale)
                ax1.fill_between(x_fill, 0, y_fill, color='blue', alpha=0.2)

                # Calculate the percentile of the comparison mean
                comparison_mean_percentile = skewnorm.cdf(meter_stats['mean'], shape, loc, scale) * 100

                # Determine energy usage category based on percentile
                if comparison_mean_percentile < 40:
                    percentile_energy_user = (f"You are a light energy user, out of 100 similar energy consumers in the {sector} benchmark dataset, "
                                            f"{comparison_mean_percentile:.0f} consume less energy. Average usage is {baseline_stats['mean']:,.3f} Wh/m^2 compared to your average usage of {meter_stats['mean']:,.3f} Wh/m^2.")
                elif comparison_mean_percentile < 60:
                    percentile_energy_user = (f"You are an average energy user, out of 100 similar energy consumers in the {sector} benchmark dataset, "
                                            f"{comparison_mean_percentile:.0f} consume less energy. Average usage is {baseline_stats['mean']:,.3f} Wh/m^2 compared to your average usage of {meter_stats['mean']:,.3f} Wh/m^2.")
                else:
                    percentile_energy_user = (f"You are a heavy energy user, out of 100 similar energy consumers in the {sector} benchmark dataset, "
                                            f"{comparison_mean_percentile:.0f} consume less energy.. Average usage is {baseline_stats['mean']:,.3f} Wh/m^2 compared to your average usage of {meter_stats['mean']:,.3f} Wh/m^2.")
                
                wrapped_text = textwrap.fill(percentile_energy_user, width=35)

                ax1.text(plt.xlim()[1]*0.61, plt.ylim()[1]*0.3, f'Percentile = {comparison_mean_percentile:.2f}%\n{wrapped_text}',fontsize=8, color='blue', ha='left', bbox=dict(facecolor='white', alpha=0.8))

                # Plot vertical lines for means and medians
                ax1.axvline(meter_stats['mean'], color='blue', linestyle='--', label=f'Comparison Mean: {meter_stats["mean"]:.2f}')

                # Add text annotations next to the lines
                ax1.text(meter_stats['mean'], ax1.get_ylim()[1]*0.75, f'Comparison\nMean = {meter_stats["mean"]:.2f}', color='blue', ha='left', bbox=dict(facecolor='white', alpha=0.8))

            # Plot important percentiles
            percentiles = [5, 25, 50, 75, 95, 99]
            plot_percentiles(ax1, percentiles, shape, loc, scale)

        except ValueError as ve:
                print(f"ValueError: {ve}")

        except scipy.stats._distn_infrastructure.FitError as fe:
                print(f"FitError: {fe}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Plot vertical lines for means and medians
        ax1.axvline(baseline_stats['mean'], color='red', linestyle='--', label=f'benchmark Mean: {baseline_stats["mean"]:.2f}')
    
        ax1.text(baseline_stats['mean'], ax1.get_ylim()[1]*0.55, f'Benchmark\nMean = {baseline_stats["mean"]:.2f}', color='red', ha='left', bbox=dict(facecolor='white', alpha=0.8))
        
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # Plot histogram for the median
        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(filtered_data['median'], bins=50, density=True, alpha=0.2, color='green', edgecolor='green')
        ax2.set_title(f'P1 Median histogram from {meter_count} sampled meters of sector: \n{sector}\nTotal area of bars in histogram equals 1 (Density=True)\nConstructed with {stack_height:,} samples')
        ax2.set_xlabel('Median (Wh/m^2, P1Norm)')
        ax2.set_ylabel('Density')

        try:
            # Fit a skewed normal distribution to the median data
            shape, loc, scale = skewnorm.fit(filtered_data['median'])

            # Generate points on the x-axis
            x = np.linspace(min(filtered_data['median']), max(filtered_data['median']), 1000)

            # Calculate the skewed normal distribution values
            pdf = skewnorm.pdf(x, shape, loc, scale)

            # Plot the skewed distribution over the histogram
            ax2.plot(x, pdf, color='black', linestyle='-', label='Skewed Normal Distribution')

            # Plot test Comparison mean if specified
            if PLOT_COMPARISON_METER:
            # Fill the area between the vertical lines and under the pdf
                x_fill = np.linspace(baseline_stats['median'], meter_stats['50%'], 1000)
                y_fill = skewnorm.pdf(x_fill, shape, loc, scale)
                ax2.fill_between(x_fill, 0, y_fill, color='blue', alpha=0.2)

                # Calculate the percentile of the comparison median
                comparison_median_percentile = skewnorm.cdf(meter_stats['50%'], shape, loc, scale) * 100
                ax2.text(meter_stats['50%'], ax2.get_ylim()[1]*0.6, f'Percentile = {comparison_median_percentile:.2f}%', color='blue', ha='left', bbox=dict(facecolor='white', alpha=0.8))
                
                # Add text annotations next to the lines
                ax2.text(meter_stats['50%'], ax2.get_ylim()[1]*0.75, f'Comparison\nMedian = {meter_stats["50%"]:.2f}', color='blue', ha='left', bbox=dict(facecolor='white', alpha=0.8))

                # Plot vertical lines for means and medians
                ax2.axvline(meter_stats['50%'], color='blue', linestyle='--', label=f'Comparison Median: {meter_stats["50%"]:.2f}')

            # Plot important percentiles
            percentiles = [5, 25, 50, 75, 95, 99]
            plot_percentiles(ax2, percentiles, shape, loc, scale)

        except ValueError as ve:
                print(f"ValueError: {ve}")

        except scipy.stats._distn_infrastructure.FitError as fe:
                print(f"FitError: {fe}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Plot vertical lines for means and medians
        ax2.axvline(baseline_stats['median'], color='green', linestyle='--', label=f'benchmark Median: {baseline_stats["median"]:.2f}')
        # Add text annotations next to the lines
        ax2.text(baseline_stats['median'], ax2.get_ylim()[1]*0.55, f'Benchmark\nMedian = {baseline_stats["median"]:.2f}', color='green', ha='left', bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # Adjust layout and display the plot
        plt.tight_layout()

        # Plot test Comparison mean if specified
        if PLOT_COMPARISON_METER:
            # Save the figure to a file
            filename = clean_filename(f"hg-{sector}-tm")
        else:
            filename = clean_filename(f"hg-{sector}")
        filepath = f'data/{filename}.png'
        plt.savefig(filepath, format='png', dpi=150)
        filepath = f'data/{filename}.svg'
        plt.savefig(filepath, format='svg')

        # Show the plot
        plt.show()

        # End by saving filtered_data to a CSV file
        filename = clean_filename(f"hg-{sector}")
        filepath = f'data/{filename}.csv'
        filtered_data.to_csv(filepath, index=False)
        
        # Append information about stack_height to the CSV file
        with open(filepath, 'a') as f:
            f.write(f"\nSamples used to generate histogram plot: {stack_height}")
    else:
        print("skipping plot because of no data!")

# Display the histogram for the specific sector
def compute_baseline_profile_data(df_filtered, meter_df, meter_stats, summary_stats_dict, loop_count,sector):
    
    # Ensure 'Getnorm' column has no zero values to avoid division by zero errors, multiply with 1000 to shift from kWh to W
    df_filtered['P1Norm'] = 1000*df_filtered['P1'] / df_filtered['Getnorm'].replace(0, np.nan)

    # Calculate summary statistics for the filtered DataFrame
    summary_stats = df_filtered.describe()

    # Check if we need to remove extreme outliers
    if REMOVE_EXTREME_OUTLIERS:

        # Calculate the Interquartile Range (IQR) bounds
        q1 = summary_stats.loc['25%']
        q3 = summary_stats.loc['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if DISPLAY_RAW_DATA:
            print("lower_bound\n", lower_bound)
            print("upper_bound\n", upper_bound)

        # Reset the index of the DataFrame
        df_filtered = df_filtered.reset_index(drop=True)
        
        # Filter outliers based on the IQR bounds
        df_filtered = df_filtered[(df_filtered['P1Norm'] >= lower_bound['P1Norm']) & (df_filtered['P1Norm'] <= upper_bound['P1Norm'])]
        
        #Remove all 0 values
        # Display the initial number of entries
        initial_count = df_filtered.shape[0]

        # Remove entries where P1 == 0 and Getnorm == 0
        df_filtered = df_filtered[(df_filtered['P1'] != 0) & (df_filtered['Getnorm'] != 0)]
        #df_filtered = df_filtered[ (df_filtered['Getnorm'] != 0)] #Using P1 != 0 may cause gaps as used above

        # Calculate the number of entries removed
        entries_removed = initial_count - df_filtered.shape[0]

        # Display raw data if flag is set
        if DISPLAY_RAW_DATA:
            print(f'Number of entries removed: {entries_removed}')
            print(df_filtered)

    # Calculate summary statistics for the filtered DataFrame after outliers has been removed
    summary_stats = df_filtered.describe()

    # Store the summary statistics in the dictionary with a unique key for each iteration
    summary_stats_dict[f'{sector}_sector'] = summary_stats
    
    if DISPLAY_RAW_DATA:
        print("Summary stats", summary_stats)

    # Combine P1 into a single stacked DataFrame
    stacked_data_p1 = df_filtered[['UnixTimestamp', 'P1Norm']].set_index('UnixTimestamp').stack().reset_index(level=1, drop=True)

    # Display raw data if flag is set
    if DISPLAY_RAW_DATA:
        print("\n\nstacked_data_p1:")
        print(stacked_data_p1)
        print(type(stacked_data_p1))
        print(stacked_data_p1.shape)

    # Calculate the amount of samples used for computation
    stack_height = stacked_data_p1.shape[0]

    # Display stack height if flag is set
    if DISPLAY_RAW_DATA:
        print(f"Stack height = {stack_height:,}")
    
    # Define quantile functions
    def quantile_05(x):
        return x.quantile(0.05)

    def quantile_25(x):
        return x.quantile(0.25)
    
    def quantile_40(x):
        return x.quantile(0.40)
    
    def quantile_60(x):
        return x.quantile(0.60)

    def quantile_75(x):
        return x.quantile(0.75)

    def quantile_95(x):
        return x.quantile(0.95)
    
    # Resample the DataFrame to 30-minute intervals and apply multiple aggregation functions
    df_resampled = stacked_data_p1.resample('30min').agg(['mean', 'median', 'std',quantile_05,quantile_25, quantile_40,quantile_60,quantile_75,quantile_95,'max', 'count'])

    # Display resampled data if flag is set
    if DISPLAY_RAW_DATA:
        print("\n\nf_resampled:")
        print(df_resampled)

    # Add DayOfWeek and TimeOfDay columns to the resampled DataFrame
    df_resampled['DayOfWeek'] = df_resampled.index.dayofweek
    df_resampled['TimeOfDay'] = df_resampled.index.time

    # Display resampled data with new columns if flag is set
    if DISPLAY_RAW_DATA:
        display(df_resampled)
    
    # Aggregate by DayOfWeek and TimeOfDay to create the benchmarkprofile
    baseline_profile = df_resampled.groupby(['DayOfWeek', 'TimeOfDay']).mean()
    baseline_stats = baseline_profile.mean()

    # Reset the index to have a flat DataFrame
    baseline_profile.reset_index(inplace=True)

    # Display benchmarkprofile and stats if flag is set
    if DISPLAY_RAW_DATA:
        print("baseline_profile:")
        print(baseline_profile)
        print(baseline_stats)

    # Return the benchmarkprofile, stack height, and benchmarkstats
    return baseline_profile, stack_height, baseline_stats, df_filtered

def compute_benchmark_plot(df_filtered, meter_df, meter_stats, baseline_profile, stack_height, baseline_stats,meter_count,sector):
    
    # Start by saving baseline_profile to a CSV file
    filename = clean_filename(f"bp-{sector}")
    filepath = f'data/{filename}.csv'
    baseline_profile.to_csv(filepath, index=False)
    
    # Append information about stack_height to the CSV file
    with open(filepath, 'a') as f:
        f.write(f"\nSamples used to generate benchmarkplot: {stack_height}")

    #--------------------------------------------------------------------------------
    #Plot Mean and standard deviation

    # Plot setup
    plt.figure(dpi=150, figsize=(14, 6))

    if DISPLAY_RAW_DATA:
    # Display baseline_profile and baseline_stats
        print("baseline_stats")
        display(baseline_stats)

    # Plot mean with label showing mean value from baseline_stats
    plt.plot(baseline_profile.index, baseline_profile['mean'], label=f'Mean: {baseline_stats["mean"]:,.3f}', color='red', alpha=0.4)
    
    # Fill between mean - std and mean + std with label showing std value from baseline_stats
    plt.fill_between(baseline_profile.index, baseline_profile['mean'] - baseline_profile['std'], baseline_profile['mean'] + baseline_profile['std'], alpha=0.2, color='red', label=f'Std Dev: {baseline_stats["std"]:,.3f}')
    
    if DISPLAY_RAW_DATA:
    # Find the index of the highest value in df_filtered['P1'] and retrieve the corresponding SNUMBER
        highest_user_index = df_filtered['P1Norm'].idxmax()
        highest_snumber = df_filtered.loc[highest_user_index, 'SNUMBER']
    
    if DISPLAY_RAW_DATA:
        # Print information about the highest user
        print(f"The SNUMBER for the highest user is: {highest_snumber} at {df_filtered['P1Norm'].max():,.3f} Wh/m^2 at index {highest_user_index:,} \n")

    if PLOT_COMPARISON_METER:
        try:
            #Get a slice of data from the comparison meter.
            meter_stats,meter_df = slice_comparison_meter(meter_df)

            # Plot test Comparison data if specified
            plt.plot(meter_df.index, meter_df['P1Norm'], color='blue', label=f"Performance Comparison: \nMean: {meter_stats['mean']:,.3f}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Add shaded area for nighttime (18:00 to 06:00)
    night_start = pd.to_datetime('18:00:00').time()
    night_end = pd.to_datetime('06:00:00').time()

    # Simulate night time by finding indices for nighttime shading
    night_indices = []
    for i in range(len(baseline_profile)):
        current_time = baseline_profile['TimeOfDay'].iloc[i]
        if night_start <= current_time or current_time <= night_end:
            night_indices.append(i)

    # Group the continuous ranges for nighttime shading
    if night_indices:
        start = night_indices[0]
        for i in range(1, len(night_indices)):
            if night_indices[i] != night_indices[i-1] + 1:
                plt.axvspan(start, night_indices[i-1], color='black', alpha=0.3)
                start = night_indices[i]
        plt.axvspan(start, night_indices[-1], color='black', alpha=0.3)

    # Map numeric day of the week to day name for labeling
    day_of_week_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Create labels by combining day names and times
    labels_day = [f"{day_of_week_mapping[dow]} {time.strftime('%H:%M')}" for dow, time in zip(baseline_profile['DayOfWeek'], baseline_profile['TimeOfDay'])]

    # Combine day of the week and time labels
    labels_hour = [time.strftime('%H:%M') for time in baseline_profile['TimeOfDay']]

    step_hour = 12
    x_ticks = np.arange(0, len(labels_hour), step_hour)

    # Function to determine label based on index, this will either print Day and time at 00:00 or just time for other values
    def get_label(i):
        if not (i % 4):
            return labels_day[i * step_hour]
        elif (i % 4):
            return labels_hour[i * step_hour]
        else:
            return ''

    # Generate labels using list comprehension
    x_labels = [get_label(i) for i in range(len(x_ticks))]
    
    # Display raw data if specified
    if DISPLAY_RAW_DATA:
        print(x_ticks)
        print(x_labels)

    # Set x-axis ticks and labels
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
    
    # Add legend, title, labels, and grid
    plt.legend()
    plt.title(f'benchmarkdata for "{sector}" Market sector using {meter_count} sampled meters, constructed with {stack_height:,} samples')
    plt.xlabel('Time')
    plt.ylabel('Power Usage (Wh/m^2, P1)')
    plt.grid(True)

    # Save the figure to a file
    if PLOT_COMPARISON_METER:
        filename = clean_filename(f"pt-mn-{sector}-tm")
    else:
        filename = clean_filename(f"pt-mn-{sector}")
    filepath = f'data/{filename}.png'
    plt.savefig(filepath, format='png', dpi=150,bbox_inches='tight')
    filepath = f'data/{filename}.svg'
    plt.savefig(filepath, format='svg',bbox_inches='tight')

    #------------------------------------------------------------------------------------------
    #Plot Median and Quantiles

    # Display the plot
    plt.show()

    # Plot setup
    plt.figure(dpi=150, figsize=(14, 6))

    if DISPLAY_RAW_DATA:
    # Display baseline_profile and baseline_stats
        print("baseline_stats")
        display(baseline_stats)
    
    # Plot median with label showing median value from baseline_stats
    plt.plot(baseline_profile.index, baseline_profile['quantile_95'], label=f'Quantile 95th: {baseline_stats["quantile_95"]:,.3f}', color='red', alpha=0.2)
    plt.plot(baseline_profile.index, baseline_profile['quantile_75'], label=f'Quantile 75th: {baseline_stats["quantile_75"]:,.3f}', color='orange', alpha=0.2)
    plt.plot(baseline_profile.index, baseline_profile['quantile_60'], label=f'Quantile 60th: {baseline_stats["quantile_60"]:,.3f}', color='yellow', alpha=0.2)
    plt.plot(baseline_profile.index, baseline_profile['median'], label=f'Median: {baseline_stats["median"]:,.3f}', color='Green', linewidth=2.5, alpha=1)
    plt.plot(baseline_profile.index, baseline_profile['quantile_40'], label=f'Quantile 40th: {baseline_stats["quantile_40"]:,.3f}', color='green', alpha=0.2)
    plt.plot(baseline_profile.index, baseline_profile['quantile_25'], label=f'Quantile 25th: {baseline_stats["quantile_25"]:,.3f}', color='olive', alpha=0.2)
    plt.plot(baseline_profile.index, baseline_profile['quantile_05'], label=f'Quantile 5th: {baseline_stats["quantile_05"]:,.3f}', color='darkgreen', alpha=0.2)

    # Fill between the quantiles
    plt.fill_between(baseline_profile.index, baseline_profile['quantile_05'], baseline_profile['quantile_25'], color='darkgreen', alpha=0.4)
    plt.fill_between(baseline_profile.index, baseline_profile['quantile_25'], baseline_profile['quantile_40'], color='green', alpha=0.2)
    plt.fill_between(baseline_profile.index, baseline_profile['quantile_40'], baseline_profile['quantile_60'], color='yellowgreen', alpha=0.2)
    plt.fill_between(baseline_profile.index, baseline_profile['quantile_60'], baseline_profile['quantile_75'], color='yellow', alpha=0.2)
    plt.fill_between(baseline_profile.index, baseline_profile['quantile_75'], baseline_profile['quantile_95'], color='orange', alpha=0.2)

    if DISPLAY_RAW_DATA:
        # Print information about the highest user
        print(f"The SNUMBER for the highest user is: {highest_snumber} at {df_filtered['P1Norm'].max():,.3f} Wh/m^2 at index {highest_user_index:,} \n")

    if PLOT_COMPARISON_METER:
        try:
            #Get a piece of comparison meter to plot against benchmark
            meter_stats,meter_df = slice_comparison_meter(meter_df)

            # Plot test Comparison data if specified
            plt.plot(meter_df.index, meter_df['P1Norm'], color='blue', label=f"Performance Comparison: \nMedian: {meter_stats['50%']:,.3f}")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Add shaded area for nighttime (18:00 to 06:00)
    night_start = pd.to_datetime('18:00:00').time()
    night_end = pd.to_datetime('06:00:00').time()

    # Simulate night time by finding indices for nighttime shading
    night_indices = []
    for i in range(len(baseline_profile)):
        current_time = baseline_profile['TimeOfDay'].iloc[i]
        if night_start <= current_time or current_time <= night_end:
            night_indices.append(i)

    # Group the continuous ranges for nighttime shading
    if night_indices:
        start = night_indices[0]
        for i in range(1, len(night_indices)):
            if night_indices[i] != night_indices[i-1] + 1:
                plt.axvspan(start, night_indices[i-1], color='black', alpha=0.3)
                start = night_indices[i]
        plt.axvspan(start, night_indices[-1], color='black', alpha=0.3)

    # Map numeric day of the week to day name for labeling
    day_of_week_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Create labels by combining day names and times
    labels_day = [f"{day_of_week_mapping[dow]} {time.strftime('%H:%M')}" for dow, time in zip(baseline_profile['DayOfWeek'], baseline_profile['TimeOfDay'])]

    # Combine day of the week and time labels
    labels_hour = [time.strftime('%H:%M') for time in baseline_profile['TimeOfDay']]

    step_hour = 12
    x_ticks = np.arange(0, len(labels_hour), step_hour)

    # Function to determine label based on index
    def get_label(i):
        if not (i % 4):
            return labels_day[i * step_hour]
        elif (i % 4):
            return labels_hour[i * step_hour]
        else:
            return ''

    # Generate labels using list comprehension
    x_labels = [get_label(i) for i in range(len(x_ticks))]
    
    # Display raw data if specified
    if DISPLAY_RAW_DATA:
        print(x_ticks)
        print(x_labels)

    # Set x-axis ticks and labels
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
    
    # Add legend, title, labels, and grid
    plt.legend()
    plt.title(f'benchmarkdata for "{sector}" Market sector using {meter_count} sampled meters, constructed with {stack_height:,} samples')
    plt.xlabel('Time')
    plt.ylabel('Power Usage (Wh/m^2, P1)')
    plt.grid(True)

    # Save the figure to a file
    if PLOT_COMPARISON_METER:
        filename = clean_filename(f"pt-md-{sector}-tm")
    else:
        filename = clean_filename(f"pt-md-{sector}")
    filepath = f'data/{filename}.png'
    plt.savefig(filepath, format='png', dpi=150,bbox_inches='tight')
    filepath = f'data/{filename}.svg'
    plt.savefig(filepath, format='svg',bbox_inches='tight')

    # Display the plot
    plt.show()

def compute_dataframe_segment(sector, df_filtered_col):
    # Filter according to the specific market sector
    df_filtered = df_filtered_col[(df_filtered_col['MarketSector'] == sector)]
    
    # Convert UnixTimestamp to datetime 
    df_filtered['UnixTimestamp'] = dd.to_datetime(df_filtered['UnixTimestamp'], unit='s', utc=True)
    
    # Convert timezone to Canada/Newfoundland (replace with your desired timezone)
    df_filtered['UnixTimestamp'] = df_filtered['UnixTimestamp'].dt.tz_convert('Canada/Newfoundland')

    # Ensure the DataFrame is sorted by UnixTimestamp
    df_filtered = df_filtered.sort_values('UnixTimestamp')

    # Get the earliest data from the dataset and set the offset to start with
    start_date = df_filtered['UnixTimestamp'].min() + pd.DateOffset(days=START_DAY, weeks=START_WEEK, months=START_MONTH)

    # Set the end date for the dataframe
    end_date = start_date + pd.DateOffset(days=END_DAY, weeks=END_WEEK, months=END_MONTH)

    # Filter the DataFrame according to the specified start and end dates
    df_filtered = df_filtered[(df_filtered['UnixTimestamp'] >= start_date) &
                              (df_filtered['UnixTimestamp'] < end_date)]

    # For debugging purposes: Display raw data and compute start/end dates
    if DISPLAY_RAW_DATA:
        display(df_filtered.head(10))  # Display first 10 rows
        display(df_filtered.tail(10))  # Display last 10 rows
        start_date = start_date.compute()  # Compute start date if using Dask
        end_date = end_date.compute()  # Compute end date if using Dask
        print(f"The start date for this graph is {start_date} up to {end_date}, "
              f"displaying a total of {end_date - start_date} days of data")
    
    #------------------------
    #Ideal method to remove P1 values where the whole set is 0 for that meter but causes issues with DASK

    # Remove all SNUMBERS where the whole P1 set = 0
    # Group by SNUMBER and calculate the sum of P1 for each group
    # grouped_P1 = df_filtered.groupby('SNUMBER')['P1'].sum()

    # Filter out SNUMBERs where the sum of P1 is 0
    # non_zero_snumber_P1 = grouped_P1[grouped_P1 != 0].index

    # # Filter the original DataFrame to keep only rows with non-zero SNUMBERs
    # df_filtered = df_filtered[df_filtered['SNUMBER'].isin(non_zero_snumber_P1)]

    #Alternative but causes gaps
    # df_filtered = df_filtered[df_filtered['P1'] != 0] //Causes gaps
    #-----------------------------

    # Remove rows where Getnorm is equal to 0, will cause divide by 0
    df_filtered = df_filtered[df_filtered['Getnorm'] != 0]
    
    
    # Compute the required data 
    df_filtered = df_filtered.compute()

    return df_filtered

#Returns the second largest number in a dataset, used for alpha calculation as largest number is too large, 2nd largest works better.
def second_largest(counts):
    unique_counts = list(set(counts))  # Remove duplicates
    if len(unique_counts) < 2:
        return None  # Not enough distinct values
    unique_counts.sort(reverse=True)
    return unique_counts[1]

def calculate_sector_specific_stats(summary_stats_dict,sorted_sector_counts):
    """
    Calculate sector-specific statistics from a dictionary of summary statistics DataFrames.

    Parameters:
    - summary_stats_dict (dict): A dictionary where keys are sector identifiers and values are summary statistics DataFrames.

    Returns:
    - None (results are written to file and printed to console)
    """
    if DISPLAY_RAW_DATA:
        print(f"\n\nSummary stats:\n{summary_stats_dict}\n\n")
        print(f"\n\nsorted_sector_counts:\n{sorted_sector_counts}\n\n")

    # Extract values and store them in separate dictionaries
    mean_dict = {key: value.loc['mean', 'P1Norm'] for key, value in summary_stats_dict.items()}
    max_dict = {key: value.loc['max', 'P1Norm'] for key, value in summary_stats_dict.items()}
    percentile_25_dict = {key: value.loc['25%', 'P1Norm'] for key, value in summary_stats_dict.items()}
    median_dict = {key: value.loc['50%', 'P1Norm'] for key, value in summary_stats_dict.items()}
    percentile_75_dict = {key: value.loc['75%', 'P1Norm'] for key, value in summary_stats_dict.items()}
    count_dict = {key: value.loc['count', 'P1Norm'] for key, value in summary_stats_dict.items()}

    # Sort the dictionaries by their values
    sorted_mean = sorted(mean_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_max = sorted(max_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_percentile_25 = sorted(percentile_25_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_median = sorted(median_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_percentile_75 = sorted(percentile_75_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_count = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)

    # Extract sector counts into a dictionary for easy access
    sector_counts_dict = sorted_sector_counts.to_dict()

    # Open a file in write mode
    with open('data\\sorted_statistics.txt', 'w') as file:
        # Write the header
        file.write("Sorting by various statistics:\n\n")
        
        # Function to write sorted values to the file
        def write_sorted_values(file, title, sorted_values):
            file.write(f"Sorting via the {title}:\n")
            for key, value in sorted_values:
                sector = key.split('_')[0]  # Extract the sector name
                count = sector_counts_dict.get(sector, 0)  # Get the count of sampled meters
                if not (title == 'Count'): #Count do not need Wh/m2 appended to the end.
                    file.write(f"{key} P1Norm: {value:,.0f} Wh/m^2, {count} sampled meters\n")
                else:
                    file.write(f"{key} P1Norm: {value:,.0f}, {count} sampled meters\n")
            file.write("\n")
        
        # Write each sorted statistic to the file
        write_sorted_values(file, "mean", sorted_mean)
        write_sorted_values(file, "max", sorted_max)
        write_sorted_values(file, "25% percentile", sorted_percentile_25)
        write_sorted_values(file, "50% median", sorted_median)
        write_sorted_values(file, "75% percentile", sorted_percentile_75)
        write_sorted_values(file, "Count", sorted_count)

    # Print sorted results to the console
    def print_sorted_values(title, sorted_values):
        print(f"\nSorting via the {title}:")
        for key, value in sorted_values:
            sector = key.split('_')[0]  # Extract the sector name
            count = sector_counts_dict.get(sector, 0)  # Get the count of sampled meters
            if not (title == 'Count'):
                print(f"{key} P1Norm: {value:,.0f} Wh/m^2, {count} sampled meters")
            else:
                print(f"{key} P1Norm: {value:,.0f}, {count} sampled meters")
    if DISPLAY_RAW_DATA:
        #Print each sorted statistic to the console using the print_sorted_values function
        print_sorted_values("mean", sorted_mean)
        print_sorted_values("max", sorted_max)
        print_sorted_values("25% percentile", sorted_percentile_25)
        print_sorted_values("50% median", sorted_median)
        print_sorted_values("75% percentile", sorted_percentile_75)
        print_sorted_values("Count", sorted_count)

    # Plotting all different sectors sorted by their energy density footprint.
    def plot_sorted_values(title, sorted_values, sector_counts_dict):
        # Filter out NaN values
        filtered_values = [(sector, value) for sector, value in sorted_values if not np.isnan(value)]
        sectors = [key.split('_')[0] for key, _ in filtered_values]
        values = [value for _, value in filtered_values]
        counts = [sector_counts_dict.get(sector, 0) for sector in sectors]
        
        if len(counts) == 0:  # Handle case where no valid data is available
            print("No valid data to plot.")
            return
        
        #Alpha closer to 1 signify more data available, lower alpha = less data fot that sector.
        min_count = min(counts)
        max_count = second_largest(counts)#max(counts) #Too large, using 2nd largest instead.
        alphas = [min((count - min_count) / (max_count - min_count) * 0.7 + 0.3, 1) for count in counts]#Ensure Alphas don't exceed 1

        # Initiate plot and size
        fig, ax = plt.subplots(figsize=(20, 6))
        bars = ax.bar(sectors, values, alpha=0.7)
        
        # Plot each sector in the bar graph
        for bar, alpha, count in zip(bars, alphas, counts):
            bar.set_alpha(alpha)
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, max_count*0.02, f'samples = {count}\nenergy density = {height:.3f} ', ha='center', va='baseline', rotation='vertical', fontsize=6, color='black')

        #Set title, labels, grid, etc
        ax.set_title(f'Sorting via the {title}')
        ax.set_xlabel('Sectors')
        ax.set_ylabel('P1Norm (Wh/m^2)')
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save and display plot
        filename = clean_filename(f"sv-{title}")
        filepath = f'data/{filename}.png'
        plt.savefig(filepath, format='png', dpi=150)
        filepath = f'data/{filename}.svg'
        plt.savefig(filepath, format='svg')
        plt.show()

    # Plot each sorted statistic
    plot_sorted_values("mean", sorted_mean, sector_counts_dict)
    plot_sorted_values("max", sorted_max, sector_counts_dict)
    plot_sorted_values("25% percentile", sorted_percentile_25, sector_counts_dict)
    plot_sorted_values("50% median", sorted_median, sector_counts_dict)
    plot_sorted_values("75% percentile", sorted_percentile_75, sector_counts_dict)

#Get a sample of data from a meter to plot against the benchmark
def slice_comparison_meter(meter_df):
    #----------------------
    #Only describe part of MeterDF that we will display on screen
    # Extract a middle chunk of meter_df for plotting
    total_rows = len(meter_df)
    start_idx = (total_rows - 336 * 3) // 2
    meter_df = meter_df.iloc[start_idx:start_idx + 336 * 3]

    # Reset the index of meter_df
    meter_df.reset_index(drop=True, inplace=True)

    # Convert UnixTimestamp to datetime if it's not already
    #meter_df['UnixTimestamp'] = pd.to_datetime(meter_df['UnixTimestamp'])#Createss a warning message, Dask do not like this
    meter_df.loc[:, 'UnixTimestamp'] = pd.to_datetime(meter_df['UnixTimestamp'])

    # Find the first Monday at 00:00:00 in meter_df
    first_monday = meter_df[(meter_df['UnixTimestamp'].dt.weekday == 0) & (meter_df['UnixTimestamp'].dt.hour == 0)].iloc[0]
    first_monday['UnixTimestamp'] = pd.to_datetime(first_monday['UnixTimestamp'])
    start_index = meter_df[meter_df['UnixTimestamp'] == first_monday['UnixTimestamp']].index[0]
    meter_df = meter_df.iloc[start_index:start_index + 336]

    # Reset the index of meter_df again
    meter_df.reset_index(drop=True, inplace=True)

    # Display raw data if specified
    if DISPLAY_RAW_DATA:
        print("\n\nmeter_df\n\n ")
        display(meter_df)
    
    # Compute descriptive statistics for 'P1' column of the selected meter
    meter_stats = meter_df['P1Norm'].describe()

    return meter_stats, meter_df

#Selects a random meter from the dataset to plot against the benchmark
def get_comparison_meter(df_filtered):
    """
    Selects a random Comparison meter from a filtered DataFrame and computes descriptive statistics.

    Parameters:
    - df_filtered (DataFrame): DataFrame containing filtered data.

    Returns:
    - meter_df (DataFrame): DataFrame filtered to include data for a random meter.
    - meter_stats (Series): Descriptive statistics (e.g., mean, std, min, max) for 'P1' column of the selected meter.
    """
    try:
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            # Randomly select a Comparison (SNUMBER) from df_filtered
            RANDOM_PERFORMANCE_COMPARISON = df_filtered.sample(n=1)['SNUMBER'].iloc[0]
            
            # Filter df_filtered to include only data for the selected Comparison (SNUMBER)
            meter_df = df_filtered[df_filtered['SNUMBER'] == RANDOM_PERFORMANCE_COMPARISON].copy()
            
            # Check if the sum of 'P1' is larger than one, avoid getting a meter with very low consumption
            if meter_df['P1'].sum() > 1:
                # Ensure 'Getnorm' column has no zero values to avoid division by zero errors,
                # multiply with 1000 to shift from kWh to Wh
                meter_df.loc[:, 'P1Norm'] = 1000 * meter_df['P1'] / meter_df['Getnorm'].replace(0, np.nan)
                break  # Exit the loop if successful
            else:
                retry_count += 1  # Increment retry counter if sum of 'P1' is zero

        # Check if the maximum retries were reached
        if retry_count == max_retries:
            print("Failed to find a valid meter after maximum retries.")
        else:
            print(f"Valid meter found after {retry_count} tries.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        # Create meter_df with the same shape but filled with zeros in case we could not get a relevant comparison meter
        meter_df = pd.DataFrame(columns=df_filtered.columns)

        # Create a dummy row with zero values
        dummy_row = pd.DataFrame([{col: 0 for col in df_filtered.columns}])

        # Append the dummy row to meter_df
        meter_df = pd.concat([meter_df, dummy_row], ignore_index=True)

        # Add new column
        meter_df['P1Norm'] = 0

    # Grabs a slice of data for the comparison meter
    meter_stats, meter_df = slice_comparison_meter(meter_df)
    
    return meter_df, meter_stats

def main():

    #Set the counter when processing limited sectors
    iteration_count_sector = MARKET_SECTOR_SAMPLE_SIZE 
    loop_count=0

    # Filter the DataFrame based on meterpoint type
    df_filtered_meter_point_type = df[(df['MeterPointType_id'] == METERPOINTTYPE_ID)]

    # Select only the columns needed for plotting
    df_filtered_col = df_filtered_meter_point_type[['AccountID','MarketSector','UnixTimestamp', 'SNUMBER', 'P1','Getnorm']]

    # Create an empty dictionary to store the summary statistics
    summary_stats_dict = {}

    # Iterate through every market sector
    for sector, count in sorted_sector_counts[SECTOR_START:SECTOR_END].items(): #Loop though defined sectors
        
        #Process loop
        loop_count+=1
        print(f"Currently processing loop {loop_count} of {min(SECTOR_END, len(sorted_sector_counts))}")

        # Stop processing when done with the loop
        if iteration_count_sector == 0:
            break  # Exit the loop

        # Decrement the counter
        iteration_count_sector -= 1

        #Show to amount of unique Accounts as well as unique meters for this sector
        meter_count = meter_counts.get(sector, 'N/A')  # Get corresponding meter_count or default to 'N/A'
        print(f"Market Sector: {sector} - Unique AccountID Count: {count}")
        print(f"Market Sector: {sector} - Unique Meter Count: {meter_count}")

        #Process a dask datafreme according to sector and filtered column
        df_filtered = compute_dataframe_segment(sector,df_filtered_col)

        #selects a random comparison meter from the specific sector
        meter_df,meter_stats = get_comparison_meter(df_filtered)

        #Compute a benchmarkprofile that will be used to performa benchmark against.
        baseline_profile ,stack_height, baseline_stats, df_filtered = compute_baseline_profile_data(df_filtered, meter_df,meter_stats,summary_stats_dict,loop_count,sector)

        # #Display a histogram of the benchmarkprofile for that sector
        compute_histogram(df_filtered,summary_stats_dict,loop_count,meter_df,meter_stats,baseline_stats,meter_count,sector,stack_height)

        # #Display a plot of the benchmarkprofile for that sector
        compute_benchmark_plot(df_filtered,meter_df,meter_stats,baseline_profile,stack_height,baseline_stats,meter_count,sector)

    #Calculate the specific statistical information for the specific sector
    calculate_sector_specific_stats(summary_stats_dict,sorted_sector_counts)

# Initialize an empty DataFrame with specified columns
sorted_sector_counts = pd.DataFrame(columns=['MarketSector', 'Count'])

DISPLAY_RAW_DATA = 0 #For debugging, shows more info around dataframes
MARKET_SECTOR_SAMPLE_SIZE = 100 #How many market sectors would you like to process? Currently there are 79, chooing a larger number selects everything

# Specify if you would like to start at a specific sector and end at another, ideal for job scheduling
SECTOR_START = 0 #0 is first sector
SECTOR_END = 40#Last sector to do

PLOT_COMPARISON_METER =1#A randomly selected power meter from that specfic sector is chosen and then plotted over the analysed data to make a comparison
REMOVE_EXTREME_OUTLIERS = 1#Apply IQR and remove those datapoints

#Specifiy the the start and end time date offset of the data that you want to include in the analysis, choose 120 months(10 years) for the whole dataset
START_DAY = 0
START_WEEK = 0
START_MONTH= 0

END_DAY=0
END_WEEK=0
END_MONTH=120

#Choose the type of meter you want to analyses
METERPOINTTYPE_ID = 3 #Primary Supply/Grid Supply

# Needed to run on windows in non-interactive mode
if __name__ == '__main__':
    
    #-------------------------------------
    #Uncomment for HPC use in a non-interactive environment.

    #set interactive mode off
    plt.ioff()

    #non-gui backend
    matplotlib.use('Agg')

    #--------------------------------------

    # Attempt to use an existing Dask client if it has been defined earlier
    try:
        client
    except NameError:
        # If 'client' is not defined, catch the NameError and start a new Dask client
        print("No dask client defined, starting client")
        # Initialize a Dask client with specific configurations:
        # - 16 workers
        # - 2 threads per worker
        # - 24GB memory limit per worker
        client = Client(n_workers=16, threads_per_worker=2, memory_limit='24GB')
    else:
        # If 'client' is already defined, print a message indicating that it's already defined
        print("Dask client already defined.")

    # Display the client details to confirm it's running
    client

    # Print the versions of the libraries being used
    print(f'Dask version: {dask.__version__}')
    print(f'Pandas version: {pd.__version__}')
    print(f'Scipy version: {scipy.__version__}')

    # Read the CSV file into a Dask DataFrame with specified data types for each column
    df = dd.read_csv(
        "/sdata/data_clean.csv",  # Complete dataset
        "/sdata/data_samplee.csv",  # Sample dataset
        dtype={
            'AccountID': 'int64',          # Account ID as integer
            'Costcentre': 'object',        # Cost centre as string (object)
            'MarketSector': 'object',      # Market sector as string (object)
            'MeterpointID': 'int64',       # Meter point ID as integer
            'MeterPointType': 'object',    # Meter point type as string (object)
            'MeterPointType_id': 'int64',  # Meter point type ID as integer
            'UnixTimestamp': 'uint64',     # Unix timestamp as unsigned integer
            'P1': 'float64',               # P1 as float
            'Q1': 'float64',               # Q1 as float
            'MeterID': 'int64',            # Meter ID as integer
            'SNUMBER': 'object',           # SNUMBER as string (object)
            'Getnorm': 'float64'           # Getnorm as float
        }
    )

    # Repartition the DataFrame to optimize memory usage and parallel processing
    # Adjust the partition size as needed. Too small partitions will increase overhead, while too few will underutilize resources.
    # Smaller size proved to work faster on local machine
    df = df.repartition(partition_size='0.1GB')

    # Print the number of partitions in the Dask DataFrame
    print(f'Number of partitions: {df.npartitions}')

    print("\nMarket sectors sorted by descending AccountID\n")

    # Filter the dataframe to only select rows where MeterPointType_id equals the specified ID
    # Also, select only the 'MarketSector' and 'AccountID' columns
    ddf = df[df['MeterPointType_id'] == METERPOINTTYPE_ID][['MarketSector', 'AccountID']]

    # Group by 'MarketSector' and count unique 'AccountID's in each sector
    sector_counts = ddf.groupby('MarketSector')['AccountID'].nunique().compute()

    # Sort the counts in descending order
    sorted_sector_counts = sector_counts.sort_values(ascending=False)

    # Print the sorted sector counts
    for sector, count in sorted_sector_counts.items():
        print(f"Market Sector: {sector} - Unique AccountID Count: {count}")

    print("\nMarket sectors sorted by descending Meters\n")

    # Filter the dataframe to only select rows where MeterPointType_id equals the specified ID
    # Also, select only the 'MarketSector' and 'SNUMBER' columns
    ddf = df[df['MeterPointType_id'] == METERPOINTTYPE_ID][['MarketSector', 'SNUMBER']]

    # Group by 'MarketSector' and count unique 'SNUMBER's in each sector
    meter_counts = ddf.groupby('MarketSector')['SNUMBER'].nunique().compute()

    # Sort the counts in descending order
    sorted_meter_counts = meter_counts.sort_values(ascending=False)

    # Print the sorted meter counts
    for sector, meter_count in sorted_meter_counts.items():
        print(f"Market Sector: {sector} - Unique Meter Count: {meter_count}")

    #In order to keep variables out of scope for the jupyter notebook, a main loop was used.
    main()