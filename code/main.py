#%%
# Import necessary libraries
import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 258N
df_258N_merge = pd.concat([df_258N_B2, df_258N_B6, df_258N_B11, df_258N_B12, df_258N_EVI, df_258N_hue], axis=1, join='outer')
# Print the first five rows of 258N_merge file
df_258N_merge.head()

# %%
# Print the shape of df_258N_merge dataframe
df_258N_merge.shape

#%%
# Assuming df is your DataFrame
print("Column Names:")
print(df_258N_merge.columns.tolist())

#%%
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 258N location
df_258N_first_checkset = df_258N_merge[['crop_id', 'crop_name', 'B6_crop_id', 'B6_crop_name', 'B11_crop_id', 'B11_crop_name',
                                         'B12_crop_id', 'B12_crop_name', 'EVI_crop_id', 'EVI_crop_name','hue_crop_id', 'hue_crop_name'  ]]
# Picking up the sample of 20 from the above dataframe
df_258N_first_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, point for all six files of 258N location
df_258N_second_checkset = df_258N_merge[['id', 'fid', 'point','B6_id', 'B6_fid', 'B6_point',                                         
                                         'B11_id', 'B11_fid', 'B11_point','B12_id', 'B12_fid', 'B12_point',                                    
                                         'EVI_id','EVI_fid', 'EVI_point','hue_id', 'hue_fid', 'hue_point']]
                                         
# Picking up the sample of 20 from the above dataframe
df_258N_second_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files of 258N location
df_258N_third_checkset = df_258N_merge[['SHAPE_AREA','SHAPE_LEN', 'B6_SHAPE_AREA','B6_SHAPE_LEN',
                                         'B11_SHAPE_AREA','B11_SHAPE_LEN','B12_SHAPE_AREA','B12_SHAPE_LEN',
                                         'EVI_SHAPE_AREA','EVI_SHAPE_LEN','hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Picking up the sample of 20 from the above dataframe
df_258N_third_checkset.sample(20)

#%%
# Dropping all the columns with only zero or single values from all files of 258N location
cols_to_drop = ['B2_ts_complexity_cid_ce','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation',
                'B6_id','B6_fid','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_point','B6_ts_complexity_cid_ce','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation',
                'B12_id','B12_fid','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_point','B12_ts_complexity_cid_ce','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation',
                'B11_id','B11_fid','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN','B11_point', 'B11_ts_complexity_cid_ce', 'B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation',
                'EVI_id','EVI_fid','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_point','EVI_ts_complexity_cid_ce','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation',
                'hue_id','hue_fid','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_point','hue_ts_complexity_cid_ce','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation',
                ]  # List of columns to drop
df_258N_merge.drop(columns=cols_to_drop, inplace=True)

# Remove columns with duplicate names
df_258N_merged = df_258N_merge.loc[:, ~df_258N_merge.columns.duplicated()]

# Print the shape of df_258N_merge dataframe after dropping the columns
df_258N_merged.shape
 
#%%
df_258N_merged.head()

# %%
# Read the six files of 259N location from the directory
df_259N_B2 = pd.read_parquet("/home/ubuntu/Capstone/data/B2_34S_19E_259N.parquet")
df_259N_B6 = pd.read_parquet("/home/ubuntu/Capstone/data/B6_34S_19E_259N.parquet")
df_259N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_259N.parquet")
df_259N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_259N.parquet")
df_259N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_259N.parquet")
df_259N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_259N.parquet")

# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 259N
df_259N_merge = pd.concat([df_259N_B2, df_259N_B6, df_259N_B11, df_259N_B12, df_259N_EVI, df_259N_hue], axis=1, join='outer')
# Print the first five rows of 259N_merge file
df_259N_merge.head()

# %%
# Print the shape of df_259N_merge file
df_259N_merge.shape

#%%
# Assuming df is your DataFrame
print("Column Names:")
print(df_259N_merge.columns.tolist())

#%%
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files of 259N location
df_259N_first_checkset = df_259N_merge[['crop_id', 'crop_name', 'B6_crop_id', 'B6_crop_name', 'B11_crop_id', 'B11_crop_name',
                                         'B12_crop_id', 'B12_crop_name', 'EVI_crop_id', 'EVI_crop_name','hue_crop_id', 'hue_crop_name'  ]]
# Picking up the sample of 20 from the above dataframe
df_259N_first_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, point for all six files of 259N location
df_259N_second_checkset = df_259N_merge[['id', 'fid', 'point','B6_id', 'B6_fid', 'B6_point',                                         
                                         'B11_id', 'B11_fid', 'B11_point','B12_id', 'B12_fid', 'B12_point',                                    
                                         'EVI_id','EVI_fid', 'EVI_point','hue_id', 'hue_fid', 'hue_point']]
                                         
# Picking up the sample of 20 from the above dataframe
df_259N_second_checkset.sample(20)

#%%
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files of 259N location
df_259N_third_checkset = df_259N_merge[['SHAPE_AREA','SHAPE_LEN', 'B6_SHAPE_AREA','B6_SHAPE_LEN',
                                         'B11_SHAPE_AREA','B11_SHAPE_LEN','B12_SHAPE_AREA','B12_SHAPE_LEN',
                                         'EVI_SHAPE_AREA','EVI_SHAPE_LEN','hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Picking up the sample of 20 from the above dataframe
df_259N_third_checkset.sample(20)

#%%
# Dropping all the columns with only zero or single values from all files of 259N location
cols_to_drop = ['B2_ts_complexity_cid_ce','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation',
                'B6_id','B6_fid','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_point','B6_ts_complexity_cid_ce','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation',
                'B12_id','B12_fid','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_point','B12_ts_complexity_cid_ce','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation',
                'B11_id','B11_fid','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN','B11_point', 'B11_ts_complexity_cid_ce', 'B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation',
                'EVI_id','EVI_fid','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_point','EVI_ts_complexity_cid_ce','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation',
                'hue_id','hue_fid','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_point','hue_ts_complexity_cid_ce','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation',
                ]  # List of columns to drop
df_259N_merge.drop(columns=cols_to_drop, inplace=True)

# Remove columns with duplicate names
df_259N_merged = df_259N_merge.loc[:, ~df_259N_merge.columns.duplicated()]

# Priting the shape of df_259N_merge dataframe after dropping the above columns
df_259N_merged.shape
 
#%%
df_259N_merged.head()

#%%
# Observing the number of records of each crop in 258N location before data cleaning
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

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

#%%
# Observing the number of records of each crop in 259N location before data cleaning
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

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

# %%
# Checking for duplicates in both df_258N_merge and df_259N_merge
# Checking for duplicates in df_258N_merge
duplicate_rows_258N = df_258N_merged.duplicated()
print("Number of duplicate rows in df_258N_merge:", duplicate_rows_258N.sum())

#%%
# Checking for duplicates in df_259N_merge
duplicate_rows_259N = df_259N_merged.duplicated()
print("Number of duplicate rows in df_259N_merge:", duplicate_rows_259N.sum())

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
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

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
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

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
# Concatenate symmetry_looking columns into a single DataFrame
symmetry_df = pd.concat([df_258N_Cleaned['B2_symmetry_looking'], df_258N_Cleaned['B6_symmetry_looking'], df_258N_Cleaned['B11_symmetry_looking'],
                         df_258N_Cleaned['B12_symmetry_looking'], df_258N_Cleaned['EVI_symmetry_looking'], df_258N_Cleaned['hue_symmetry_looking']],
                        axis=1)
custom_colors = ['#013220','#005C29',  '#004E00','#228B22', '#90EE90', '#92D050', '#FFFF00']
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
                         df_259N_Cleaned['B12_symmetry_looking'], df_259N_Cleaned['EVI_symmetry_looking'], df_259N_Cleaned['hue_symmetry_looking']],
                        axis=1)
custom_colors = ['#013220','#005C29',  '#004E00','#228B22', '#90EE90', '#92D050', '#FFFF00']
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
# Specify the filename of the parquet file
filename = 'df_258N_Cleaned.parquet'

# Get the current working directory
current_directory = os.getcwd()

# Concatenate the current directory with the filename to get the full path
full_path = os.path.join(current_directory, filename)

print("Full path of the parquet file:", full_path)


#%%
# Assuming you have loaded your dataset into a DataFrame called 'df_258N_Cleaned'
# 'target_variable' is the name of your target variable
# Make sure to replace these with your actual variable names

# Filter out non-numeric columns
numeric_columns = df_258N_Cleaned.select_dtypes(include=[np.number])

# Calculate correlation
correlation = numeric_columns.corrwith(df_258N_Cleaned['crop_id'])

# Absolute correlation values
correlation = correlation.abs()

# Sort correlation values in descending order
correlation = correlation.sort_values(ascending=False)

# Display all rows and columns without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the sorted correlation values
print(correlation) 

#%%
# Assuming df is your DataFrame containing the column 'crop_name' and 'B2_standard_deviation'

# Plot boxplot for B2_standard_deviation column grouped by crop_name
plt.figure(figsize=(10, 6))
sns.boxplot(x='crop_name', y='hue_maximum', data=df_258N_Cleaned)
plt.title('B12_quantile_q_0.05 for Each Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('hue_maximum')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# %%
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Assuming X_train and y_train are your features and target variable respectively
# Separate features and target variable
X_train_258N = df_258N_Cleaned.drop(columns=['crop_name'])
y_train_258N = df_258N_Cleaned['crop_name']

# Create an instance of RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

# Perform undersampling
X_train_resampled_258N, y_train_resampled_258N = undersampler.fit_resample(X_train_258N, y_train_258N)

# Now X_train_resampled and y_train_resampled contain the balanced dataset after undersampling

# %%
# Assuming you have already performed undersampling and stored the results in X_train_resampled
# Print the shape of X_train_resampled to see the number of samples and features
print("Shape of X_train_resampled:", X_train_resampled_258N.shape)

#%%
# Optionally, you can print the first few rows of X_train_resampled
print("First few rows of X_train_resampled:")
X_train_resampled_258N.head()

#%%
# Observing the number of records of each crop in 258N location aFter Under Sampling
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = X_train_resampled_258N['crop_id'].value_counts()

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

# %%
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Assuming X_train and y_train are your features and target variable respectively
# Separate features and target variable
X_train_259N = df_259N_Cleaned.drop(columns=['crop_name'])
y_train_259N = df_259N_Cleaned['crop_name']

# Create an instance of RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)

# Perform undersampling
X_train_resampled_259N, y_train_resampled_259N = undersampler.fit_resample(X_train_259N, y_train_259N)

# Now X_train_resampled and y_train_resampled contain the balanced dataset after undersampling

# %%
# Assuming you have already performed undersampling and stored the results in X_train_resampled
# Print the shape of X_train_resampled to see the number of samples and features
print("Shape of X_train_resampled:", X_train_resampled_259N.shape)

#%%
# Optionally, you can print the first few rows of X_train_resampled
print("First few rows of X_train_resampled:")
X_train_resampled_259N.head()

#%%
# Observing the number of records of each crop in 258N location aFter Under Sampling
# Define custom green colors
custom_colors = ['#013220', '#005C29', '#004E00', '#228B22', '#90EE90']

# Group the DataFrame by the crop variable and count the number of rows for each group
crop_counts = X_train_resampled_259N['crop_id'].value_counts()

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

# %%
from sklearn.preprocessing import StandardScaler

# Assuming X_train_resampled is your resampled feature data

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data to compute mean and standard deviation
scaler.fit(X_train_resampled_258N)

# Transform your data using the computed mean and standard deviation
X_train_resampled_scaled_258N = scaler.transform(X_train_resampled_258N)

# Now X_train_resampled_scaled contains the standardized feature data

# Assuming X_train_resampled_scaled is your standardized feature data

# Print the shape of X_train_resampled_scaled
print("Shape of X_train_resampled_scaled:", X_train_resampled_scaled_258N.shape)

# Optionally, print the first few rows of X_train_resampled_scaled
print("First few rows of X_train_resampled_scaled:")
print(X_train_resampled_scaled_258N[:5])  # Print the first 5 rows


# %%
from sklearn.preprocessing import StandardScaler

# Assuming X_train_resampled is your resampled feature data
# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data to compute mean and standard deviation
scaler.fit(X_train_resampled_259N)

# Transform your data using the computed mean and standard deviation
X_train_resampled_scaled_259N = scaler.transform(X_train_resampled_259N)

# Assuming X_train_resampled_scaled is your standardized feature data

# Print the shape of X_train_resampled_scaled
print("Shape of X_train_resampled_scaled:", X_train_resampled_scaled_259N.shape)

# Optionally, print the first few rows of X_train_resampled_scaled
print("First few rows of X_train_resampled_scaled:")
print(X_train_resampled_scaled_259N[:5])  # Print the first 5 rows


# Now X_train_resampled_scaled contains the standardized feature data
# %%
