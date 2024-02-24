#%%
# Import necessary libraries
import os
import pyarrow.parquet as pq
import pandas as pd
import dask.dataframe as dd

#%%
# Define the path to the directory containing Parquet files in your Google Drive
parquet_directory = '/home/ubuntu/Capstone/data'

# Get the list of Parquet files in the directory
parquet_files = [file for file in os.listdir(parquet_directory) if file.endswith('.parquet')]
print(parquet_files)

# %%
# Read each Parquet file and do something with it (e.g., print schema)
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

# %%
# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 259N
# Read the 259N_B2 file
df_259N_B2 = pd.read_parquet("/home/ubuntu/Capstone/data/B2_34S_19E_259N.parquet")
# Print the first five rows of 259N_B2 file
df_259N_B2.head()

# %%
# Print the shape of 259N_B2 file
df_259N_B2.shape

# %%
# Read the 259N_B6 file
df_259N_B6 = pd.read_parquet("/home/ubuntu/Capstone/data/B6_34S_19E_259N.parquet")
# Print the first five rows of 259N_B6 file
df_259N_B6.head()

#%%
# Read the 259N_B11 file
df_259N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_259N.parquet")
# Print the first five rows of 259N_B11 file
df_259N_B11.head()

# %%
# Read the 259N_B12 file
df_259N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_259N.parquet")
# Print the first five rows of 259N_B12 file
df_259N_B12.head()

# %%
# Read the 259N_EVI file
df_259N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_259N.parquet")
# Print the first five rows of 259N_EVI file
df_259N_EVI.head()

# %%
# Read the 259N_hue file
df_259N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_259N.parquet")
# Print the first five rows of 259N_hue file
df_259N_hue.head()


#%%
# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 259N
df_259N_merge = pd.concat([df_259N_B2, df_259N_B6, df_259N_B11, df_259N_B12, df_259N_EVI, df_259N_hue], axis=1, join='outer')
# Print the first five rows of 259N_merge file
df_259N_merge.head()

# %%
# Print the shape of 259N_B2 file
df_259N_merge.shape



# %%
#
#
#
# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 258N
# Read the 258N_B2 file
df_258N_B2 = pd.read_parquet("/home/ubuntu/Capstone/data/B2_34S_19E_258N.parquet")
# Print the first five rows of 258N_B2 file
df_258N_B2.head()

# %%
# Print the shape of 258N_B2 file
df_258N_B2.shape

# %%
# Read the 258N_B6 file
df_258N_B6 = pd.read_parquet("/home/ubuntu/Capstone/data/B6_34S_19E_258N.parquet")
# Print the first five rows of 258N_B6 file
df_258N_B6.head()

#%%
# Read the 258N_B11 file
df_258N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_258N.parquet")
# Print the first five rows of 258N_B11 file
df_258N_B11.head()

# %%
# Read the 258N_B12 file
df_258N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_258N.parquet")
# Print the first five rows of 258N_B12 file
df_258N_B12.head()

# %%
# Read the 258N_EVI file
df_258N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_258N.parquet")
# Print the first five rows of 258N_EVI file
df_258N_EVI.head()

# %%
# Read the 258N_hue file
df_258N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_258N.parquet")
# Print the first five rows of 258N_hue file
df_258N_hue.head()


#%%
# Joining of B2, B6, B11, B12, EVI, Hue datasets of location 258N
df_258N_merge = pd.concat([df_258N_B2, df_258N_B6, df_258N_B11, df_258N_B12, df_258N_EVI, df_258N_hue], axis=1, join='outer')
# Print the first five rows of 258N_merge file
df_258N_merge.head()

# %%
# Print the shape of 258N_B2 file
df_258N_merge.shape

# %%
