# Harnessing Big Data and High Performance Compute to Understand Commercial Energy Usage and Utility Generation Dynamics

The focus of this Independent Study Project (ISP) revolves around analyzing electricity power consumption and generation. The primary objective is to examine consumption patterns among commercial and retail users. This project is motivated by the observation that many consumers pay their electricity bills without considering the trends and insights embedded within their usage data.

By analyzing these patterns, the project aims to empower consumers to make more informed decisions about their energy usage, contributing to efforts against climate change. Uncovering nuanced patterns within electricity data is expected to offer valuable insights, ultimately helping society become more energy-efficient. The passion for power generation and consumption stems from its pervasive impact on modern economies and daily life, where reliable electricity is a fundamental necessity.

## Files in This Project

<pre>
Files in this project
|->README.md [This file]
|->Energy_data_exploration.ipynb [A Jupyter notebook file that delves into the dataset, doing basic analyses and exploration]
|->Energy_data_processing.ipynb [A Jupyter notebook file that reads and computes the dataset with dask dataframes. Statistical data and images is generated and saved]
|->power_consumption_data.7z [A small sample of the dataset, full dataset does not fit in github]
|->Siku_script.sh [The script used to execute the python code contained in Energy_data_processing.ipynb]
|->.gitattributes [Required for Gitgub Large File System used by Energy_data_processing.ipynb]
|-data->bp-*.csv [Statistical data generated for a benchmark base plot]
|     ->hg-*.csv [Statistical data generated for a benchmark histogram plot]
|-images-> pt-mn-*-tm.svg [Benchmark plot with mean statistical data with a random comparison meter]
        -> pt-md-*-tm.svg [Benchmark plot with median statistical data with a random comparison meter]
        -> hg-*-tm.svg [Benchmark plot with histogram statistical data with a random comparison meter]
        -> sv-*.svg [Sorted Benchmark sector via mean, max and 25th, 50th, 75th percentile]
</pre>

## `Energy_data_exploration.ipynb`
Basic exploratory and data analysis performed on a smaller subset of the dataset. Basic information from the dataset is displayed, such as a sample, columns and data types of columns, as well as basic calculations.

### Analyses and Exploration Done
- Extraction of unique values from identifier columns `Costcentre` and `MarketSector`
- Counting unique `AccountID` coupled with each market sector
- Extracting and displaying unique `MeterPoint` attributes
- Displaying unique `Account IDs`
- Collecting and combining data for unique `Account IDs`
- Filtering and analyzing data for a specific account
- Analyzing meter point details for a selected account
- Filtering and displaying data for specific meter IDs
- Filtering data for a specified time period
- Visualizing data for meter IDs
- Plotting data for grid supply meters

## `Energy_data_processing.ipynb`
This Jupyter notebook is dedicated to the data processing of CSV files containing power data. It serves as the main module for analyzing and visualizing the given data.

### Feature Engineering
- Data Aggregation and Filtering
- Histogram and Distribution Plotting
- Time-Series Benchmark Plotting
- Statistical Data Export
  - **Features**
    - Histogram with skewed normal distribution
    - Time-Series benchmark plot
