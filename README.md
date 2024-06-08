# Harnessing Big Data and High Performance Compute to Understand Commercial Energy Usage and Utility Generation Dynamics

The focus of this Independent Study Project (ISP) revolves around analyzing electricity power consumption and generation. The primary objective is to examine consumption patterns among commercial and retail users and evaluate generation from three key sources: utility, solar, and diesel generators. This project is motivated by the observation that many consumers pay their electricity bills without considering the trends and insights embedded within their usage data.

By analyzing these patterns, the project aims to empower consumers to make more informed decisions about their energy usage, contributing to efforts against climate change. Uncovering nuanced patterns within electricity data is expected to offer valuable insights, ultimately helping society become more energy-efficient. The passion for power generation and consumption stems from its pervasive impact on modern economies and daily life, where reliable electricity is a fundamental necessity.

# Power Measurements Dataset

This dataset contains high frequency interval (30 minutes) power measurements in the retail and industrial sector.

## Columns of Interest

- **AccountID** (4301 unique values): The customer’s account, i.e., 4301 unique customer accounts.
- **Costcentre** (15 unique categories): Billing allocation according to power source or load. Categorizes what the load or supply is.
- **MeterpointID** (10,848 unique values): Virtual power meter.
- **MeterpointType** (25 unique values): Energy Consumption Categories or Metered Energy Usage, e.g., Grid, Generator, Solar, refrigeration, extractor fan, etc.
- **MarketSector** (79 unique values): Market sector that the consumer is operating in.
- **Meterpoint_ID** (10,281 unique values): Virtual power meter. This column is linked to multiple Meter_IDs.
- **Meter_ID** (10,627 unique values): Multiple Meter_IDs can be aggregated to a single Meterpoint_ID.
- **RDATE**: The 30-minute interval that the power measurement has been done. For example, “2024-01-01 10:00:00” indicates a reading on January 1, 2024, at 10 AM.
- **P1**: The real power in kWh (kilowatt hour) consumed or generated for that 30-minute interval.
- **Q1**: The imaginary power in kVArh (kilo reactive volt-amp hour) consumed or generated for that 30-minute interval.
