#%%
# Import necessary libraries
import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

#%%
# Define the path to the directory containing Parquet files
parquet_directory = '/home/ubuntu/Capstone/data'

# Get the list of Parquet files in the directory
parquet_files = [file for file in os.listdir(parquet_directory) if file.endswith('.parquet')]
print(parquet_files)

# %%
## Data Preprocessing

# Read each Parquet file and print the schema of all variables in the parquet files
for file_name in parquet_files:
    file_path = os.path.join(parquet_directory, file_name)
    table = pq.read_table(file_path)
    print(f"Schema for {file_name}:")
    print(table.schema)

#%%
# Initialize an empty dictionary to store null value counts for each column
null_counts = {}

# Iterate through each Parquet file in the directory
for filename in os.listdir(parquet_directory):
    if filename.endswith('.parquet'):
        # Read Parquet file into DataFrame
        df = pd.read_parquet(os.path.join(parquet_directory, filename))

        # Calculate null value counts for each column in the DataFrame
        null_counts[filename] = df.isnull().sum()

# Print null value counts for each column in each Parquet file
for filename, counts in null_counts.items():
    print(f"Null value counts for {filename}:")
    print(counts)
    print()  # Print empty line for better readability

# %%
# Initialize an empty dictionary to store unique value counts for each column
unique_counts = {}

# Iterate through each Parquet file in the directory
for filename in os.listdir(parquet_directory):
    if filename.endswith('.parquet'):
        # Read Parquet file into DataFrame
        df = pd.read_parquet(os.path.join(parquet_directory, filename))

        # Calculate unique value counts for each column in the DataFrame
        unique_counts[filename] = df.nunique()

# Print unique value counts for each column in each Parquet file
for filename, counts in unique_counts.items():
    print(f"Unique value counts for {filename}:")
    print(counts)
    print()  # Print empty line for better readability


#%%
# Renames the common column names of 'B6', 'B11', 'B12', 'EVI', 'hue' files of both 258 and 259 locations
# Define the list of files to process
list_of_files = ['B6', 'B11', 'B12', 'EVI', 'hue']

# Iterate over the files in the directory
for filename in os.listdir(parquet_directory):
    if filename.endswith('.parquet') and not filename.startswith('B2'):
        # Load the DataFrame from the parquet file
        df = pd.read_parquet(os.path.join(parquet_directory, filename))
        
        # Extract the prefix from the filename
        prefix = filename.split('_')[0]
        
        # Check if the prefix is in the list of files to process
        if prefix in list_of_files:
            # Define the column name mapping for the current prefix
            column_name_mapping = {
                'point': f'{prefix}_point',
                'crop_id': f'{prefix}_crop_id',
                'crop_name': f'{prefix}_crop_name',
                'id': f'{prefix}_id',
                'fid': f'{prefix}_fid',
                'SHAPE_AREA': f'{prefix}_SHAPE_AREA',
                'SHAPE_LEN': f'{prefix}_SHAPE_LEN'
                # Add more mappings as needed
            }
            
            # Rename columns based on the mapping
            df.rename(columns=column_name_mapping, inplace=True)
            
            # Save the DataFrame back to the parquet file
            df.to_parquet(os.path.join(parquet_directory, filename))

# %%
# Read the six files of 258N location from the directory
df_258N_B2 = pd.read_parquet("/home/ubuntu/Capstone/data/B2_34S_19E_258N.parquet")
df_258N_B6 = pd.read_parquet("/home/ubuntu/Capstone/data/B6_34S_19E_258N.parquet")
df_258N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_258N.parquet")
df_258N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_258N.parquet")
df_258N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_258N.parquet")
df_258N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_258N.parquet")

# Joining of B2, B6, B11, B12, EVI, and Hue datasets of location 258N
# df_258N_merge = pd.concat([df_258N_B2, df_258N_B6, df_258N_B11, df_258N_B12, df_258N_EVI, df_258N_hue], axis=1, join='outer')

# Joining of B2, B6, B11, B12, and Hue datasets of location 258N
df_258N_merge = pd.concat([df_258N_B2, df_258N_B6, df_258N_B11, df_258N_B12, df_258N_hue], axis=1, join='outer')

# Print the first five rows of 258N_merge file
df_258N_merge.shape

#%%
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 258N location
df_258N_first_checkset = df_258N_merge[['crop_id', 'crop_name', 'B6_crop_id', 'B6_crop_name', 'B11_crop_id', 'B11_crop_name',
                                         'B12_crop_id', 'B12_crop_name','hue_crop_id', 'hue_crop_name']]
# Picking up the sample of 20 from the above dataframe
df_258N_first_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, point for all six files of 258N location
df_258N_second_checkset = df_258N_merge[['id', 'fid', 'point','B6_id', 'B6_fid', 'B6_point',                                         
                                         'B11_id', 'B11_fid', 'B11_point','B12_id', 'B12_fid', 'B12_point',                                    
                                         'hue_id', 'hue_fid', 'hue_point']]
                                         
# Picking up the sample of 20 from the above dataframe
df_258N_second_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files of 258N location
df_258N_third_checkset = df_258N_merge[['SHAPE_AREA','SHAPE_LEN', 'B6_SHAPE_AREA','B6_SHAPE_LEN',
                                         'B11_SHAPE_AREA','B11_SHAPE_LEN','B12_SHAPE_AREA','B12_SHAPE_LEN',
                                         'hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Picking up the sample of 20 from the above dataframe
df_258N_third_checkset.sample(20)

#%%
# # Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 258N location
df_258N_fourth_checkset = df_258N_merge[[ 'B2_count_above_mean', 'B2_count_below_mean','B6_count_above_mean', 'B6_count_below_mean', 'B11_count_above_mean','B11_count_below_mean', 'B12_count_above_mean','B12_count_below_mean','hue_count_above_mean','hue_count_below_mean',]]
# Picking up the sample of 20 from the above dataframe
df_258N_fourth_checkset.sample(20)


#%%
# Dropping all the columns with only zero/single values or repitative from all files of 258N location
cols_to_drop = ['id', 'fid', 'point','crop_id', 'SHAPE_AREA','SHAPE_LEN', 'B2_ts_complexity_cid_ce','B2_count_below_mean','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation','B2_ratio_beyond_r_sigma_r_1','B2_ratio_beyond_r_sigma_r_2',
                'B6_id','B6_fid','B6_point','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_ts_complexity_cid_ce','B6_count_below_mean','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation','B6_ratio_beyond_r_sigma_r_1','B6_ratio_beyond_r_sigma_r_2',
                'B12_id','B12_fid','B12_point','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_ts_complexity_cid_ce','B12_count_below_mean','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation','B12_ratio_beyond_r_sigma_r_1','B12_ratio_beyond_r_sigma_r_2',
                'B11_id','B11_fid','B11_point','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN', 'B11_ts_complexity_cid_ce', 'B11_count_below_mean','B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation','B11_ratio_beyond_r_sigma_r_1','B11_ratio_beyond_r_sigma_r_2',
                'hue_id','hue_fid','hue_point','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_ts_complexity_cid_ce','hue_count_below_mean','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation','hue_ratio_beyond_r_sigma_r_1','hue_ratio_beyond_r_sigma_r_2',
                ]  # List of columns to drop
                #'EVI_id','EVI_fid','EVI_point','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_ts_complexity_cid_ce','EVI_count_below_mean','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation','EVI_ratio_beyond_r_sigma_r_1','EVI_ratio_beyond_r_sigma_r_2',
df_258N_merge.drop(columns=cols_to_drop, inplace=True)

# Remove columns with duplicate names
df_258N_merged = df_258N_merge.loc[:, ~df_258N_merge.columns.duplicated()]

# Print the shape of df_258N_merge dataframe after dropping the columns
df_258N_merged.shape

#%%
# Assuming df_258N_merged is your existing DataFrame
# Define functions to calculate vegetation indices

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red)

def calculate_ndwi(nir, swir1):
    return (nir - swir1) / (nir + swir1)

def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def calculate_vrn(red, nir):
    return red / (nir + red)

def calculate_savi(nir, red, l=0.5):
    return (1+l)* (nir - red) / (nir + red + l)

def calculate_nbr2(swir2, swir1):
    return (swir2 - swir1) / (swir2 + swir1)

def calculate_mirbi(swir2, swir1):
    return (10 * swir2) -(9.8 * swir1) + 2

# def calculate_msi(nir, red):
#     return (nir - 1.5 * red + 0.5) / np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red)) - 0.5)

# def calculate_crop_yield(b6_mean, shape_area, b12_std):
#     return b6_mean + shape_area + b12_std

# def calculate_vhi(b11_mean, b11_std, b2_median):
#     return b11_mean + b11_std - b2_median

# def calculate_temporal_trend(b11_autocorr_lag1, b12_autocorr_lag2):
#     return b11_autocorr_lag1 - b12_autocorr_lag2

# def calculate_anomaly_score(b6_abs_sum_changes, b11_complexity):
#     return b6_abs_sum_changes + b11_complexity

# def calculate_water_probability(b2_std, b6_large_std):
#     return b2_std + b6_large_std

# Apply the functions to the DataFrame to create new columns
df_258N_merged['NDVI'] = calculate_ndvi(df_258N_merged['B6_mean'], df_258N_merged['B2_mean'])
df_258N_merged['NDWI'] = calculate_ndwi(df_258N_merged['B6_mean'], df_258N_merged['B11_mean'])
df_258N_merged['EVI'] = calculate_evi(df_258N_merged['B6_mean'], df_258N_merged['B2_mean'], df_258N_merged['hue_mean'])
df_258N_merged['VRN'] = calculate_vrn(df_258N_merged['B2_mean'], df_258N_merged['B6_mean'])
df_258N_merged['SAVI'] = calculate_savi(df_258N_merged['B6_mean'], df_258N_merged['B2_mean'])
df_258N_merged['NBR2'] = calculate_nbr2(df_258N_merged['B12_mean'],df_258N_merged['B11_mean'])
df_258N_merged['MIRBI'] = calculate_mirbi(df_258N_merged['B12_mean'],df_258N_merged['B11_mean'])

# df_258N_merged['MSI'] = calculate_msi(df_258N_merged['B6_standard_deviation'], df_258N_merged['hue_mean'])
# df_258N_merged['VHI'] = calculate_vhi(df_258N_merged['B11_mean'], df_258N_merged['B11_standard_deviation'], df_258N_merged['B2_median'])
# df_258N_merged['Water_Probability'] = calculate_water_probability(df_258N_merged['B2_standard_deviation'], df_258N_merged['B6_standard_deviation'])
# df_258N_merged['Temporal_Trend'] = calculate_temporal_trend(df_258N_merged['B11_autocorr_lag_1'], df_258N_merged['B12_autocorr_lag_2'])
# df_258N_merged['Crop_Yield'] = calculate_crop_yield(df_258N_Cleaned['B6_mean'], df_258N_Cleaned['SHAPE_AREA'], df_258N_Cleaned['B12_standard_deviation'])
# df_258N_Cleaned['Anomaly_Score'] = calculate_anomaly_score(df_258N_Cleaned['B6_absolute_sum_of_changes'], df_258N_Cleaned['B11_ts_complexity_cid_ce'])

df_258N_merged.shape

#%%
# Observing the number of records of each crop in 258N location before data cleaning
# Define custom Green Colors
# custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = df_258N_merged['crop_name'].value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(crop_counts.index, crop_counts.values, color=custom_colors)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Crop ID')
plt.ylabel('Number of Records')
plt.title('Histogram of Records per each Crop in 258N location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# Checking for duplicates in both df_258N_merge and df_259N_merge

# Checking for duplicates in df_258N_merge
duplicate_rows_258N = df_258N_merged.duplicated()
print("Number of duplicate rows in df_258N_merge:", duplicate_rows_258N.sum())

# Dropping duplicate rows from df_258N_merge
df_258N_merged = df_258N_merged.drop_duplicates()

# Now, df_258N_merge contains only unique rows

#%%
df_258N_merged.shape

# %%
# Checking for outliers using  IQR in both df_258N_merge and df_259N_merge

# Checking for outliers using IQR in df_258N_merge
# Filter out columns with string data types
numeric_columns = df_258N_merged.select_dtypes(include=['number'])

# Calculate the first quartile (Q1) and third quartile (Q3) for numeric columns
Q1 = numeric_columns.quantile(0.05)
Q3 = numeric_columns.quantile(0.95)

# Calculate the interquartile range (IQR) for numeric columns
IQR = Q3 - Q1

# Define the outlier bounds for numeric columns
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Check for outliers in numeric columns
outliers = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound)).any(axis=1)

# Print the number of outliers
print("Number of outliers in df_258N_merge:", outliers.sum())

#%%
# Remove the outliers from df_258N_merge 
df_258N_Cleaned = df_258N_merged[~outliers]
# Print the shape of df_258N_Cleaned dataframe
df_258N_Cleaned.shape

#%%
# Observing the number of records of each crop in 258N location after data cleaning
# Define custom Green Colors
# custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = df_258N_Cleaned['crop_name'].value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(crop_counts.index, crop_counts.values, color=custom_colors)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Crop ID')
plt.ylabel('Number of Records')
plt.title('Histogram of Records(Cleaned) per each Crop in 258N location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# Print the Top 5 rows of df_258N_merged Dataset
df_258N_Cleaned.head()

# %%
# Read the six files of 259N location from the directory
df_259N_B2 = pd.read_parquet("/home/ubuntu/Capstone/data/B2_34S_19E_259N.parquet")
df_259N_B6 = pd.read_parquet("/home/ubuntu/Capstone/data/B6_34S_19E_259N.parquet")
df_259N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_259N.parquet")
df_259N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_259N.parquet")
df_259N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_259N.parquet")
df_259N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_259N.parquet")

# Joining of B2, B6, B11, B12, EVI and Hue datasets of location 259N
# df_259N_merge = pd.concat([df_259N_B2, df_259N_B6, df_259N_B11, df_259N_B12, df_259N_EVI, df_259N_hue], axis=1, join='outer')

# Joining of B2, B6, B11, B12, and Hue datasets of location 259N
df_259N_merge = pd.concat([df_259N_B2, df_259N_B6, df_259N_B11, df_259N_B12, df_259N_hue], axis=1, join='outer')

# Print the first five rows of 259N_merge file
df_259N_merge.shape

#%%
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 259N location
df_259N_first_checkset = df_259N_merge[['crop_id', 'crop_name', 'B6_crop_id', 'B6_crop_name', 'B11_crop_id', 'B11_crop_name',
                                         'B12_crop_id', 'B12_crop_name','hue_crop_id', 'hue_crop_name'  ]]
# Picking up the sample of 20 from the above dataframe
df_259N_first_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, point for all six files of 259N location
df_259N_second_checkset = df_259N_merge[['id', 'fid', 'point','B6_id', 'B6_fid', 'B6_point',                                         
                                         'B11_id', 'B11_fid', 'B11_point','B12_id', 'B12_fid', 'B12_point',                                    
                                       'hue_id', 'hue_fid', 'hue_point']]
                                         
# Picking up the sample of 20 from the above dataframe
df_259N_second_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files of 259N location
df_259N_third_checkset = df_259N_merge[['SHAPE_AREA','SHAPE_LEN', 'B6_SHAPE_AREA','B6_SHAPE_LEN',
                                         'B11_SHAPE_AREA','B11_SHAPE_LEN','B12_SHAPE_AREA','B12_SHAPE_LEN',
                                        'hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Picking up the sample of 20 from the above dataframe
df_259N_third_checkset.sample(20)

#%%
# # Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 258N location
df_259N_fourth_checkset = df_259N_merge[[ 'B2_count_above_mean', 'B2_count_below_mean','B6_count_above_mean', 'B6_count_below_mean', 'B11_count_above_mean','B11_count_below_mean', 'B12_count_above_mean','B12_count_below_mean','hue_count_above_mean','hue_count_below_mean',]]
# Picking up the sample of 20 from the above dataframe
df_259N_fourth_checkset.sample(20)

#%%
# Dropping all the columns with only zero or single values from all files of 259N location
cols_to_drop = ['id', 'fid', 'point','crop_id', 'SHAPE_AREA','SHAPE_LEN', 'B2_ts_complexity_cid_ce','B2_count_below_mean','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation', 'B2_ratio_beyond_r_sigma_r_1','B2_ratio_beyond_r_sigma_r_2',
                'B6_id','B6_fid','B6_point','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_ts_complexity_cid_ce','B6_count_below_mean','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation','B6_ratio_beyond_r_sigma_r_1','B6_ratio_beyond_r_sigma_r_2',
                'B12_id','B12_fid','B12_point','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_ts_complexity_cid_ce','B12_count_below_mean','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation','B12_ratio_beyond_r_sigma_r_1','B12_ratio_beyond_r_sigma_r_2',
                'B11_id','B11_fid','B11_point','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN', 'B11_ts_complexity_cid_ce', 'B11_count_below_mean','B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation','B11_ratio_beyond_r_sigma_r_1','B11_ratio_beyond_r_sigma_r_2',
                'hue_id','hue_fid','hue_point','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_ts_complexity_cid_ce','hue_count_below_mean','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation','hue_ratio_beyond_r_sigma_r_1','hue_ratio_beyond_r_sigma_r_2',
                ]  # List of columns to drop
                #'EVI_id','EVI_fid','EVI_point','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_ts_complexity_cid_ce','EVI_count_below_mean','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation','EVI_ratio_beyond_r_sigma_r_1','EVI_ratio_beyond_r_sigma_r_2',

df_259N_merge.drop(columns=cols_to_drop, inplace=True)

# Remove columns with duplicate names
df_259N_merged = df_259N_merge.loc[:, ~df_259N_merge.columns.duplicated()]

# Priting the shape of df_259N_merge dataframe after dropping the above columns
df_259N_merged.shape

#%%
# Observing the number of records of each crop in 259N location before data cleaning
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = df_259N_merged['crop_name'].value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(crop_counts.index, crop_counts.values, color=custom_colors)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Crop ID')
plt.ylabel('Number of Records')
plt.title('Histogram of Records per each Crop in 259N location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# Assuming df_259N_merged is your existing DataFrame
# Define functions to calculate vegetation indices

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red)

def calculate_ndwi(nir, swir1):
    return (nir - swir1) / (nir + swir1)

def calculate_evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def calculate_vrn(red, nir):
    return red / (nir + red)

def calculate_savi(nir, red, l=0.5):
    return (1+l) * (nir - red) / (nir + red + l)

def calculate_nbr2(swir2, swir1):
    return (swir2 - swir1) / (swir2 + swir1)

def calculate_mirbi(swir2, swir1):
    return (10 * swir2) -(9.8 * swir1) + 2

# def calculate_msi(nir, red):
#     return (nir - 1.5 * red + 0.5) / np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * np.sqrt(red)) - 0.5)

# def calculate_crop_yield(b6_mean, shape_area, b12_std):
#     return b6_mean + shape_area + b12_std

# def calculate_vhi(b11_mean, b11_std, b2_median):
#     return b11_mean + b11_std - b2_median

# def calculate_temporal_trend(b11_autocorr_lag1, b12_autocorr_lag2):
#     return b11_autocorr_lag1 - b12_autocorr_lag2

# def calculate_anomaly_score(b6_abs_sum_changes, b11_complexity):
#     return b6_abs_sum_changes + b11_complexity

# def calculate_water_probability(b2_std, b6_large_std):
#     return b2_std + b6_large_std

# Apply the functions to the DataFrame to create new columns
df_259N_merged['NDVI'] = calculate_ndvi(df_259N_merged['B6_mean'], df_259N_merged['B2_mean'])
df_259N_merged['NDWI'] = calculate_ndwi(df_259N_merged['B6_mean'], df_259N_merged['B11_mean'])
df_259N_merged['EVI'] = calculate_evi(df_259N_merged['B6_mean'], df_259N_merged['B2_mean'], df_259N_merged['hue_mean'])
df_259N_merged['VRN'] = calculate_vrn(df_259N_merged['hue_mean'], df_259N_merged['B6_mean'])
df_259N_merged['SAVI'] = calculate_savi(df_259N_merged['B6_mean'], df_259N_merged['B2_mean'])
df_259N_merged['NBR2'] = calculate_nbr2(df_259N_merged['B12_mean'],df_259N_merged['B11_mean'])
df_259N_merged['MIRBI'] = calculate_mirbi(df_259N_merged['B12_mean'],df_259N_merged['B11_mean'])

# df_259N_merged['MSI'] = calculate_msi(df_259N_merged['B6_standard_deviation'], df_259N_merged['hue_standard_deviation'])
# df_259N_merged['VHI'] = calculate_vhi(df_259N_merged['B11_mean'], df_259N_merged['B11_standard_deviation'], df_259N_merged['B2_median'])
# df_259N_merged['Temporal_Trend'] = calculate_temporal_trend(df_259N_merged['B11_autocorr_lag_1'], df_259N_merged['B12_autocorr_lag_2'])
# df_259N_merged['Water_Probability'] = calculate_water_probability(df_259N_merged['B2_standard_deviation'], df_259N_merged['B6_standard_deviation'])
# df_259N_Cleaned['Crop_Yield'] = calculate_crop_yield(df_259N_Cleaned['B6_mean'], df_259N_Cleaned['SHAPE_AREA'], df_259N_Cleaned['B12_standard_deviation'])
# df_258N_Cleaned['Anomaly_Score'] = calculate_anomaly_score(df_258N_Cleaned['B6_absolute_sum_of_changes'], df_258N_Cleaned['B11_ts_complexity_cid_ce'])

# # Display the concatenated DataFrame
# #print(df_258N_Cleaned
print(df_259N_merged.shape)

#%%
# Checking for duplicates in df_259N_merge
duplicate_rows_259N = df_259N_merged.duplicated()
print("Number of duplicate rows in df_259N_merge:", duplicate_rows_259N.sum())

# Dropping duplicate rows from df_258N_merge
df_259N_merged = df_259N_merged.drop_duplicates()


# %%
# Checking for outliers using IQR in df_259N_merge
# Filter out columns with string data types
numeric_columns = df_259N_merged.select_dtypes(include=['number'])

# Calculate the first quartile (Q1) and third quartile (Q3) for numeric columns
Q1 = numeric_columns.quantile(0.05)
Q3 = numeric_columns.quantile(0.95)

# Calculate the interquartile range (IQR) for numeric columns
IQR = Q3 - Q1

# Define the outlier bounds for numeric columns
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Check for outliers in numeric columns
outliers = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound)).any(axis=1)

# Print the number of outliers
print("Number of outliers in df_259N_merge:", outliers.sum())

#%%
# Remove the outliers from df_259N_merge 
df_259N_Cleaned = df_259N_merged[~outliers]
# Print the shape of df_259N_Cleaned dataframe
df_259N_Cleaned.shape

#%%
# Observing the number of records of each crop in 259N location after data cleaning
# Define custom green colors
# custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = df_259N_Cleaned['crop_name'].value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(crop_counts.index, crop_counts.values, color=custom_colors)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Crop ID')
plt.ylabel('Number of Records')
plt.title('Histogram of Records(Cleaned) per each Crop in 259N location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
df_259N_merged.head()

#%%
# Exploratory Data Analysis

#%%
# Concatenate symmetry_looking columns into a single DataFrame
symmetry_df = pd.concat([df_258N_Cleaned['B2_symmetry_looking'], df_258N_Cleaned['B6_symmetry_looking'], df_258N_Cleaned['B11_symmetry_looking'],
                         df_258N_Cleaned['B12_symmetry_looking'], df_258N_Cleaned['hue_symmetry_looking']],
                        axis=1)
# Define custom Green Colors
# custom_colors = ['#013220','#005C29',  '#004E00','#228B22', '#90EE90', '#92D050', '#FFFF00']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']
# Plot the stacked bar plot
symmetry_df.apply(pd.Series.value_counts).plot(kind='bar', stacked=True, figsize=(10, 6), color=custom_colors)
plt.title('Stacked Bar Plot of Symmetry Looking by Band in df_258N_Cleaned')
plt.xlabel('Symmetry Looking')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Band')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# Concatenate symmetry_looking columns into a single DataFrame
symmetry_df = pd.concat([df_259N_Cleaned['B2_symmetry_looking'], df_259N_Cleaned['B6_symmetry_looking'], df_259N_Cleaned['B11_symmetry_looking'],
                         df_259N_Cleaned['B12_symmetry_looking'], df_259N_Cleaned['hue_symmetry_looking']],
                        axis=1)
# Define custom Green Colors
# custom_colors = ['#013220','#005C29',  '#004E00','#228B22', '#90EE90', '#92D050', '#FFFF00']
# Define custom Blue Colors
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Plot the stacked bar plot
symmetry_df.apply(pd.Series.value_counts).plot(kind='bar', stacked=True, figsize=(10, 6),color=custom_colors)
plt.title('Stacked Bar Plot of Symmetry Looking by Band i df_259N_merge')
plt.xlabel('Symmetry Looking')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Band')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# # Assuming df is your DataFrame containing the column 'crop_name' and 'B2_standard_deviation'

# # Plot boxplot for B2_standard_deviation column grouped by crop_name
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='crop_name', y='B2_abs_energy', data=df_258N_Cleaned)
# plt.title('SHAPE_LEN')
# plt.xlabel('Crop Type')
# plt.ylabel('SHAPE_LEN')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# %%
# Under Sampling
# Perform Under Sampling on 258N location

# Assuming X_train and y_train are your features and target variable respectively
# Separate features and target variable
X_258 = df_258N_Cleaned.drop(columns=['crop_name'])
y_258 = df_258N_Cleaned['crop_name']

# Create an instance of RandomUnderSampler
undersampler_258 = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

# Perform undersampling
X_258_undersampled, y_258_undersampled = undersampler_258.fit_resample(X_258,y_258)

# Now X_train_resampled and y_train_resampled contain the balanced dataset after undersampling

# Print the shape of X_train_resampled to see the number of samples and features
print("Shape of X_258_undersampled:", X_258_undersampled.shape)

# %%
#%%
# Observing the number of records of each crop in 258N location after data cleaning
# Define custom green colors
#custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']
custom_colors = ['#000080', '#87CEEB','#4682B4', '#1E90FF', '#6A5ACD']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = y_258_undersampled.value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(crop_counts.index, crop_counts.values, color=custom_colors)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Crop ID')
plt.ylabel('Number of Records')
plt.title('Histogram of Records(Cleaned) per each Crop in 259N location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# Standardize the Sampled Data of 258N before performing modeling
# Assuming X_258_undersampled is your resampled feature data

# Initialize StandardScaler
scaler_258 = StandardScaler()

# Fit the scaler to your data to compute mean and standard deviation
scaler_258.fit(X_258_undersampled)

# Transform your data using the computed mean and standard deviation
X_scaled_258N = scaler_258.transform(X_258_undersampled)

# Now X_train_resampled_scaled contains the standardized feature data

# Assuming X_train_resampled_scaled is your standardized feature data

# Print the shape of X_train_resampled_scaled
print("Shape of X_train_resampled_scaled:", X_scaled_258N.shape)

#%%
# Divide the 258N data into train and test data
X_train_258N, X_test_258N, y_train_258N, y_test_258N = train_test_split(X_scaled_258N, y_258_undersampled, test_size=0.20, random_state=42)
clf_258N = RandomForestClassifier(n_estimators=200,max_depth=20, criterion='entropy', min_samples_split=15, n_jobs=-1, random_state=0,)
clf_258N.fit(X_train_258N,y_train_258N)

#%%
# Predict on the sampled test set
y_pred_258N = clf_258N.predict(X_test_258N)

# Generate classification report for sampled test set
report_sampled_258N = classification_report(y_test_258N, y_pred_258N)
print("Classification Report on Location 258 using Train & Test Split:\n", report_sampled_258N)

#%%
# Extract feature importances
feature_importances_258 = clf_258N.feature_importances_

# Create a list of tuples containing feature names and their importances
feature_importance_list_258 = list(zip(X_258_undersampled.columns, feature_importances_258))

# Sort the feature importance list by importance score in descending order
sorted_feature_importance_258 = sorted(feature_importance_list_258, key=lambda x: x[1], reverse=True)
print(sorted_feature_importance_258)

# Select the top 20 features
top_features_258 = [feature[0] for feature in sorted_feature_importance_258[:70]]
print(top_features_258)

#%%
# Calculate the sum of feature importances for the top features
sum_top_importance_258 = sum(feature[1] for feature in sorted_feature_importance_258[:70])

# Print the sum
print("Sum of top features' importance:", sum_top_importance_258)

#%%
import matplotlib.pyplot as plt

# Extract feature names and importances
features = [feature[0] for feature in sorted_feature_importance_258]
importances = [feature[1] for feature in sorted_feature_importance_258]

# Plotting
plt.figure(figsize=(50, 20))
plt.barh(features, importances, color='#1F456E')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to display highest importance at the top
plt.show()

# %%
# # Define the list of variables you want to check
variables_of_interest = ["NDVI", "EVI", "NDWI","VRN","SAVI","NBR2","MIRBI"]
# Initialize a dictionary to store the positions of importance for each variable
positions_and_importance = {}

# Search for each variable in the sorted feature importance list
for variable in variables_of_interest:
    found = False
    for index, (feature, importance) in enumerate(sorted_feature_importance_258):
        if feature == variable:
            positions_and_importance[variable] = {"position": index + 1, "importance": importance}
            found = True
            break
    if not found:
        positions_and_importance[variable] = {"position": "Not found", "importance": None}

# Print the positions and importance for each variable
for variable, data  in positions_and_importance.items():
    print(f"The position of importance for {variable} is: {data['position']} and importance is: {data['importance']} ")

# %%
# Under Sampling
# Perform Under Sampling on 258N location

# Assuming X_train and y_train are your features and target variable respectively
# Separate features and target variable
X_258_top = df_258N_Cleaned[top_features_258].copy()
#X_258_80 = X_258 = df_258N_Cleaned.drop(columns=['crop_name'])
y_258_top = df_258N_Cleaned['crop_name']

# Create an instance of RandomUnderSampler
undersampler_258_top = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

# Perform undersampling
X_258_undersampled_top, y_258_undersampled_top = undersampler_258_top.fit_resample(X_258_top,y_258_top)

# Now X_train_resampled and y_train_resampled contain the balanced dataset after undersampling

# Print the shape of X_train_resampled to see the number of samples and features
print("Shape of X_258_undersampled_top:", X_258_undersampled_top.shape)

# %%
# Standardize the Sampled Data of 258N before performing modeling
# Assuming X_258_undersampled is your resampled feature data

# Initialize StandardScaler
scaler_258_top = StandardScaler()

# Fit the scaler to your data to compute mean and standard deviation
scaler_258_top.fit(X_258_undersampled_top)

# Transform your data using the computed mean and standard deviation
X_scaled_258N_top = scaler_258_top.transform(X_258_undersampled_top)

# Now X_train_resampled_scaled contains the standardized feature data

# Assuming X_train_resampled_scaled is your standardized feature data

# Print the shape of X_train_resampled_scaled
print("Shape of X_train_scaled_top:", X_scaled_258N_top.shape)

# %%
# Divide the 258N data into train and test data

#X_train_258N_top_RF, X_test_258N_top_RF, y_train_258N_top_RF, y_test_258N_top_RF = train_test_split(X_scaled_258N_top, y_258_undersampled_top, test_size=0.20, random_state=42)
X_train_258N_top_RF, y_train_258N_top_RF = X_scaled_258N_top, y_258_undersampled_top
clf_258N_top_RF = RandomForestClassifier(n_estimators=200,max_depth=15, criterion='entropy', min_samples_split=15, n_jobs=-1, random_state=0,)
clf_258N_top_RF.fit(X_train_258N_top_RF,y_train_258N_top_RF)

#%%
# Assuming X_train and y_train are your features and target variable respectively
# Separate features and target variable
X_259_top = df_259N_Cleaned[top_features_258].copy()
y_259_top = df_259N_Cleaned['crop_name']

# Create an instance of RandomUnderSampler
undersampler_259_top = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

# Perform undersampling
X_259_undersampled_top, y_259_undersampled_top = undersampler_259_top.fit_resample(X_259_top,y_259_top)

#%%

# Initialize StandardScaler
scaler_259_top = StandardScaler()

# Fit the scaler to your data to compute mean and standard deviation
scaler_259_top.fit(X_259_undersampled_top)

# Transform your data using the computed mean and standard deviation
X_scaled_259N_top = scaler_259_top.transform(X_259_undersampled_top)

#%%
# Predict on the sampled test set
y_pred_259N_top_RF = clf_258N_top_RF.predict(X_scaled_259N_top)

# Generate classification report for sampled test set
report_sampled_259N_top_RF = classification_report(y_259_undersampled_top, y_pred_259N_top_RF)
print("Classification Report on Location 259 using Train & Test Split:\n", report_sampled_259N_top_RF)

#%%
# XGBoost

from sklearn.preprocessing import StandardScaler, LabelEncoder
# Standardize the data
scaler_89_XG = StandardScaler()

X_scaled_258N_top_XG = scaler_89_XG.fit_transform(X_258_undersampled_top)
X_scaled_259N_top_XG = scaler_89_XG.fit_transform(X_259_undersampled_top)

# Encode the target variable
label_encoder_top_258_XG = LabelEncoder()
label_encoder_top_259_XG = LabelEncoder()

y_encoded_258N_top_XG = label_encoder_top_258_XG.fit_transform(y_258_undersampled_top)
y_encoded_259N_top_XG = label_encoder_top_259_XG.fit_transform(y_259_undersampled_top)

# %%
X_train_258N_top_XG, y_train_258N_top_XG = X_scaled_258N_top_XG, y_encoded_258N_top_XG
X_test_259N_top_XG, y_test_259N_top_XG = X_scaled_259N_top_XG, y_encoded_259N_top_XG

#%%
# Divide the 258N data into train and test data

#X_train_258N_top_RF, X_test_258N_top_RF, y_train_258N_top_RF, y_test_258N_top_RF = train_test_split(X_scaled_258N_top, y_258_undersampled_top, test_size=0.20, random_state=42)
import xgboost as xgb

xgb_258N_top_XGB = xgb.XGBClassifier(n_estimators=200,max_depth=25, learning_rate=0.01,colsample_bytree=0.1,gamma=30, n_jobs=-1, random_state=42)
xgb_258N_top_XGB.fit(X_train_258N_top_XG, y_train_258N_top_XG)

#%%
# Encode the target variable
# label_encoder_top_XGB = LabelEncoder()
# Make predictions
y_pred_259N_top_XGB = xgb_258N_top_XGB.predict(X_test_259N_top_XG)

# # Decode the predicted labels
# y_pred_decoded_259N_89_XGB = label_encoder_top_XGB.inverse_transform(y_pred_259N_top_XGB)
# y_test_decoded_259N_89_XGB = label_encoder_top_XGB.inverse_transform(y_test_259N_top_XG)

# Generate classification report for sampled test set
#report_259N_89_XGB = classification_report(y_test_decoded_259N_89_XGB,y_pred_decoded_259N_89_XGB)
report_259N_89_XGB = classification_report(y_test_259N_top_XG,y_pred_259N_top_XGB)
print("Classification Report on Sampled Test Set:\n", report_259N_89_XGB)

#%%
#Light GBM

# Standardize the data
scaler_89_lgbm = StandardScaler()

X_scaled_258N_top_lgbm = scaler_89_lgbm.fit_transform(X_258_undersampled_top)
X_scaled_259N_top_lgbm = scaler_89_lgbm.fit_transform(X_259_undersampled_top)

# Encode the target variable
label_encoder_top_258_lgbm = LabelEncoder()
label_encoder_top_259_lgbm = LabelEncoder()

y_encoded_258N_top_lgbm = label_encoder_top_258_XG.fit_transform(y_258_undersampled_top)
y_encoded_259N_top_lgbm = label_encoder_top_259_XG.fit_transform(y_259_undersampled_top)

#%%
#Using entire 258 for Train and 259 locations for Test & No Train & Test Split
X_train_258N_89_lgbm, y_train_258N_89_lgbm = X_scaled_258N_top_lgbm, y_encoded_258N_top_lgbm
X_test_259N_89_lgbm, y_test_259N_89_lgbm = X_scaled_259N_top_lgbm, y_encoded_259N_top_lgbm

#%%
import lightgbm as lgb

lgb_classifier = lgb.LGBMClassifier()

# Define LightGBM classifier with the best hyperparameters
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'num_boost_round': 200, 'reg_lambda': 0.8, 'reg_alpha': 0.8, 'objective': 'multiclass',
    'num_class': 5, 'force_col_wise':'true'}
lgb_classifier_258_89_lgbm = lgb.LGBMClassifier(**best_params)

# Train the model on the entire training dataset
lgb_classifier_258_89_lgbm.fit(X_train_258N_89_lgbm, y_train_258N_89_lgbm)

#%%
# Make predictions
y_pred_259N_89_lgbm = lgb_classifier_258_89_lgbm.predict(X_test_259N_89_lgbm)

# Generate classification report for sampled test set
report_259N_89_lgbm = classification_report(y_test_259N_89_lgbm,y_pred_259N_89_lgbm)
print("Classification Report on Sampled Test Set:\n", report_259N_89_lgbm)

