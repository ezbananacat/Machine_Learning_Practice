import pandas as pd

# Load the CSV file
file_path = "data/air_pollution_death.csv"  # Replace with your actual file path
output_file = "air_pollution_deaths_filtered.csv"  # Name of the output CSV file

# Read the CSV
df = pd.read_csv(file_path)

# Apply filtering conditions
filtered_df = df[
    (df["SpatialDimValueCode"] == "CHN") &
    (df["Period"] == 2018) &
    (df["Dim1"] == "Both sexes")
]

# Select relevant columns
columns_to_keep = ["IndicatorCode", "Indicator", "ValueType", "SpatialDimValueCode", "Period", "Dim1", "Value"]
filtered_df = filtered_df[columns_to_keep]

# Save to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
