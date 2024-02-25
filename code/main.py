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

#Rename the below columns in B6 file to remove after merging 258N files
B6_new_column_names = {
    'point': 'B6_point',
    'crop_id': 'B6_crop_id',
    'crop_name': 'B6_crop_name',
    'id':'B6_id',
    'fid':'B6_fid',
    'SHAPE_AREA':'B6_SHAPE_AREA',
    'SHAPE_LEN':'B6_SHAPE_LEN'
    # Add more mappings as needed
}
df_258N_B6.rename(columns=B6_new_column_names, inplace=True)

# Print the first five rows of 258N_B6 file
df_258N_B6.head()

#%%
# Read the 258N_B11 file
df_258N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_258N.parquet")

#Rename the below columns in B11 file to remove after merging 258N files
B11_new_column_names = {
    'point': 'B11_point',
    'crop_id': 'B11_crop_id',
    'crop_name': 'B11_crop_name',
    'id':'B11_id',
    'fid':'B11_fid',
    'SHAPE_AREA':'B11_SHAPE_AREA',
    'SHAPE_LEN':'B11_SHAPE_LEN'
    # Add more mappings as needed
}
df_258N_B11.rename(columns=B11_new_column_names, inplace=True)

# Print the first five rows of 258N_B11 file
df_258N_B11.head()

# %%
# Read the 258N_B12 file
df_258N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_258N.parquet")

#Rename the below columns in B12 file to remove after merging 258N files
B12_new_column_names = {
    'point': 'B12_point',
    'crop_id': 'B12_crop_id',
    'crop_name': 'B12_crop_name',
    'id':'B12_id',
    'fid':'B12_fid',
    'SHAPE_AREA':'B12_SHAPE_AREA',
    'SHAPE_LEN':'B12_SHAPE_LEN'
    # Add more mappings as needed
}
df_258N_B12.rename(columns=B12_new_column_names, inplace=True)

# Print the first five rows of 258N_B12 file
df_258N_B12.head()

# %%
# Read the 258N_EVI file
df_258N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_258N.parquet")

#Rename the below columns in EVI file to remove after merging 258N files
EVI_new_column_names = {
    'point': 'EVI_point',
    'crop_id': 'EVI_crop_id',
    'crop_name': 'EVI_crop_name',
    'id':'EVI_id',
    'fid':'EVI_fid',
    'SHAPE_AREA':'EVI_SHAPE_AREA',
    'SHAPE_LEN':'EVI_SHAPE_LEN'
    # Add more mappings as needed
}
df_258N_EVI.rename(columns=EVI_new_column_names, inplace=True)

# Print the first five rows of 258N_EVI file
df_258N_EVI.head()

# %%
# Read the 258N_hue file
df_258N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_258N.parquet")

#Rename the below columns in hue file to remove after merging 258N files
hue_new_column_names = {
    'point': 'hue_point',
    'crop_id': 'hue_crop_id',
    'crop_name': 'hue_crop_name',
    'id':'hue_id',
    'fid':'hue_fid',
    'SHAPE_AREA':'hue_SHAPE_AREA',
    'SHAPE_LEN':'hue_SHAPE_LEN'
    # Add more mappings as needed
}
df_258N_hue.rename(columns=hue_new_column_names, inplace=True)

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

#%%
import pandas as pd
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files
# Assuming df_258N_B2, df_258N_B6, ..., df_258N_hue are your DataFrames

# Select 'crop_id' and 'crop_name' columns from each DataFrame
df_258N_B2_crop = df_258N_B2[['crop_id', 'crop_name']]
df_258N_B6_crop = df_258N_B6[['B6_crop_id', 'B6_crop_name']]
df_258N_B11_crop = df_258N_B11[['B11_crop_id', 'B11_crop_name']]
df_258N_B12_crop = df_258N_B12[['B12_crop_id', 'B12_crop_name']]
df_258N_EVI_crop = df_258N_EVI[['EVI_crop_id', 'EVI_crop_name']]
df_258N_hue_crop = df_258N_hue[['hue_crop_id', 'hue_crop_name']]
# Repeat this for the other DataFrames as well

# Concatenate the selected columns
df_258N_merge_crop = pd.concat([df_258N_B2_crop, df_258N_B6_crop, df_258N_B11_crop, df_258N_B12_crop, df_258N_B2_crop, df_258N_EVI_crop,  df_258N_hue_crop], axis=1, join='outer')

df_258N_merge_crop.sample(10)

#%%
import pandas as pd
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files

# Assuming df_258N_B2, df_258N_B6, ..., df_258N_hue are your DataFrames


# Select 'crop_id' and 'crop_name' columns from each DataFrame
df_258N_B2_crop = df_258N_B2[['id', 'fid', 'SHAPE_AREA','SHAPE_LEN']]
df_258N_B6_crop = df_258N_B6[['B6_id', 'B6_fid', 'B6_SHAPE_AREA','B6_SHAPE_LEN']]
df_258N_B11_crop = df_258N_B11[['B11_id', 'B11_fid', 'B11_SHAPE_AREA','B11_SHAPE_LEN']]
df_258N_B12_crop = df_258N_B12[['B12_id', 'B12_fid', 'B12_SHAPE_AREA','B12_SHAPE_LEN']]
df_258N_EVI_crop = df_258N_EVI[['EVI_id','EVI_fid', 'EVI_SHAPE_AREA','EVI_SHAPE_LEN']]
df_258N_hue_crop = df_258N_hue[['hue_id', 'hue_fid', 'hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Repeat this for the other DataFrames as well

# Concatenate the selected columns
df_258N_merge_crop = pd.concat([df_258N_B2_crop, df_258N_B6_crop, df_258N_B11_crop, df_258N_B12_crop, df_258N_B2_crop, df_258N_EVI_crop,  df_258N_hue_crop], axis=1, join='outer')

df_258N_merge_crop.sample(20)

#%%
# Dropping all the columns with only zero or single values from all files
cols_to_drop = ['B2_ts_complexity_cid_ce','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation',
                'B6_id','B6_fid','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_point','B6_ts_complexity_cid_ce','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation',
                'B12_id','B12_fid','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_point','B12_ts_complexity_cid_ce','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation',
                'B11_id','B11_fid','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN','B11_point', 'B11_ts_complexity_cid_ce', 'B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation',
                'EVI_id','EVI_fid','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_point','EVI_ts_complexity_cid_ce','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation',
                'hue_id','hue_fid','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_point','hue_ts_complexity_cid_ce','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation',
                ]  # List of columns to drop
df_258N_merge.drop(columns=cols_to_drop, inplace=True)

df_258N_merge.shape
 
#%%
df_258N_merge.head()



# %%
#
#
#
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

#Rename the below columns in B6 file to remove after merging 259N files
B6_new_column_names = {
    'point': 'B6_point',
    'crop_id': 'B6_crop_id',
    'crop_name': 'B6_crop_name',
    'id':'B6_id',
    'fid':'B6_fid',
    'SHAPE_AREA':'B6_SHAPE_AREA',
    'SHAPE_LEN':'B6_SHAPE_LEN'
    # Add more mappings as needed
}
df_259N_B6.rename(columns=B6_new_column_names, inplace=True)

# Print the first five rows of 259N_B6 file
df_259N_B6.head()



#%%
# Read the 259N_B11 file
df_259N_B11 = pd.read_parquet("/home/ubuntu/Capstone/data/B11_34S_19E_259N.parquet")

#Rename the below columns in B11 file to remove after merging 259N files
df_259N_B11.rename(columns={'point': 'B11_point'}, inplace=True)
df_259N_B11.rename(columns={'crop_id': 'B11_crop_id'}, inplace=True)
df_259N_B11.rename(columns={'crop_name': 'B11_crop_name'}, inplace=True)
df_259N_B11.rename(columns={'id': 'B11_id'}, inplace=True)
df_259N_B11.rename(columns={'fid': 'B11_fid'}, inplace=True)
df_259N_B11.rename(columns={'SHAPE_AREA': 'B11_SHAPE_AREA'}, inplace=True)
df_259N_B11.rename(columns={'SHAPE_LEN': 'B11_SHAPE_LEN'}, inplace=True)

# Print the first five rows of 259N_B11 file
df_259N_B11.head()

# %%
# Read the 259N_B12 file
df_259N_B12 = pd.read_parquet("/home/ubuntu/Capstone/data/B12_34S_19E_259N.parquet")

#Rename the below columns in B12 file to remove after merging 259N files
df_259N_B12.rename(columns={'point': 'B12_point'}, inplace=True)
df_259N_B12.rename(columns={'crop_id': 'B12_crop_id'}, inplace=True)
df_259N_B12.rename(columns={'crop_name': 'B12_crop_name'}, inplace=True)
df_259N_B12.rename(columns={'id': 'B12_id'}, inplace=True)
df_259N_B12.rename(columns={'fid': 'B12_fid'}, inplace=True)
df_259N_B12.rename(columns={'SHAPE_AREA': 'B12_SHAPE_AREA'}, inplace=True)
df_259N_B12.rename(columns={'SHAPE_LEN': 'B12_SHAPE_LEN'}, inplace=True)

# Print the first five rows of 259N_B12 file
df_259N_B12.head()

# %%
# Read the 259N_EVI file
df_259N_EVI = pd.read_parquet("/home/ubuntu/Capstone/data/EVI_34S_19E_259N.parquet")

#Rename the below columns in EVI file to remove after merging 259N files
df_259N_EVI.rename(columns={'point': 'EVI_point'}, inplace=True)
df_259N_EVI.rename(columns={'crop_id': 'EVI_crop_id'}, inplace=True)
df_259N_EVI.rename(columns={'crop_name': 'EVI_crop_name'}, inplace=True)
df_259N_EVI.rename(columns={'id': 'EVI_id'}, inplace=True)
df_259N_EVI.rename(columns={'fid': 'EVI_fid'}, inplace=True)
df_259N_EVI.rename(columns={'SHAPE_AREA': 'EVI_SHAPE_AREA'}, inplace=True)
df_259N_EVI.rename(columns={'SHAPE_LEN': 'EVI_SHAPE_LEN'}, inplace=True)

# Print the first five rows of 259N_EVI file
df_259N_EVI.head()

# %%
# Read the 259N_hue file
df_259N_hue = pd.read_parquet("/home/ubuntu/Capstone/data/hue_34S_19E_259N.parquet")

#Rename the below columns in hue file to remove after merging 259N files
df_259N_hue.rename(columns={'point': 'hue_point'}, inplace=True)
df_259N_hue.rename(columns={'crop_id': 'hue_crop_id'}, inplace=True)
df_259N_hue.rename(columns={'crop_name': 'hue_crop_name'}, inplace=True)
df_259N_hue.rename(columns={'id': 'hue_id'}, inplace=True)
df_259N_hue.rename(columns={'fid': 'hue_fid'}, inplace=True)
df_259N_hue.rename(columns={'SHAPE_AREA': 'hue_SHAPE_AREA'}, inplace=True)
df_259N_hue.rename(columns={'SHAPE_LEN': 'hue_SHAPE_LEN'}, inplace=True)

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

#%%
import pandas as pd
# Checking for the repitition of the existing columns like crop_id and crop_name for all six files
# Assuming df_259N_B2, df_259N_B6, ..., df_259N_hue are your DataFrames

# Select 'crop_id' and 'crop_name' columns from each DataFrame
df_259N_B2_crop = df_259N_B2[['crop_id', 'crop_name']]
df_259N_B6_crop = df_259N_B6[['B6_crop_id', 'B6_crop_name']]
df_259N_B11_crop = df_259N_B11[['B11_crop_id', 'B11_crop_name']]
df_259N_B12_crop = df_259N_B12[['B12_crop_id', 'B12_crop_name']]
df_259N_EVI_crop = df_259N_EVI[['EVI_crop_id', 'EVI_crop_name']]
df_259N_hue_crop = df_259N_hue[['hue_crop_id', 'hue_crop_name']]
# Repeat this for the other DataFrames as well

# Concatenate the selected columns
df_259N_merge_crop = pd.concat([df_259N_B2_crop, df_259N_B6_crop, df_259N_B11_crop, df_259N_B12_crop, df_259N_B2_crop, df_259N_EVI_crop,  df_259N_hue_crop], axis=1, join='outer')

df_259N_merge_crop.sample(10)

#%%
import pandas as pd
#Checking for the repitition of the existing columns like id, fid, SHAPE_AREA, SHAPE_LEN for all six files

# Assuming df_259N_B2, df_259N_B6, ..., df_259N_hue are your DataFrames

# Select 'crop_id' and 'crop_name' columns from each DataFrame
df_259N_B2_crop = df_259N_B2[['id', 'fid', 'SHAPE_AREA','SHAPE_LEN']]
df_259N_B6_crop = df_259N_B6[['B6_id', 'B6_fid', 'B6_SHAPE_AREA','B6_SHAPE_LEN']]
df_259N_B11_crop = df_259N_B11[['B11_id', 'B11_fid', 'B11_SHAPE_AREA','B11_SHAPE_LEN']]
df_259N_B12_crop = df_259N_B12[['B12_id', 'B12_fid', 'B12_SHAPE_AREA','B12_SHAPE_LEN']]
df_259N_EVI_crop = df_259N_EVI[['EVI_id','EVI_fid', 'EVI_SHAPE_AREA','EVI_SHAPE_LEN']]
df_259N_hue_crop = df_259N_hue[['hue_id', 'hue_fid', 'hue_SHAPE_AREA','hue_SHAPE_LEN']]
# Repeat this for the other DataFrames as well

# Concatenate the selected columns
df_259N_merge_crop = pd.concat([df_259N_B2_crop, df_259N_B6_crop, df_259N_B11_crop, df_259N_B12_crop, df_259N_B2_crop, df_259N_EVI_crop,  df_259N_hue_crop], axis=1, join='outer')

df_259N_merge_crop.sample(20)

#%%
# Dropping all the columns with only zero or single values from all files
cols_to_drop = ['B2_ts_complexity_cid_ce','B2_doy_of_maximum_dates','B2_doy_of_minimum_dates','B2_large_standard_deviation','B2_variance_larger_than_standard_deviation',
                'B6_id','B6_fid','B6_crop_id','B6_crop_name','B6_SHAPE_AREA','B6_SHAPE_LEN','B6_point','B6_ts_complexity_cid_ce','B6_doy_of_maximum_dates','B6_doy_of_minimum_dates','B6_large_standard_deviation','B6_variance_larger_than_standard_deviation',
                'B12_id','B12_fid','B12_crop_id','B12_crop_name','B12_SHAPE_AREA','B12_SHAPE_LEN','B12_point','B12_ts_complexity_cid_ce','B12_doy_of_maximum_dates','B12_doy_of_minimum_dates','B12_large_standard_deviation','B12_variance_larger_than_standard_deviation',
                'B11_id','B11_fid','B11_crop_id','B11_crop_name','B11_SHAPE_AREA','B11_SHAPE_LEN','B11_point', 'B11_ts_complexity_cid_ce', 'B11_doy_of_maximum_dates','B11_doy_of_minimum_dates','B11_large_standard_deviation','B11_variance_larger_than_standard_deviation',
                'EVI_id','EVI_fid','EVI_crop_id','EVI_crop_name','EVI_SHAPE_AREA','EVI_SHAPE_LEN','EVI_point','EVI_ts_complexity_cid_ce','EVI_doy_of_maximum_dates','EVI_doy_of_minimum_dates','EVI_large_standard_deviation','EVI_variance_larger_than_standard_deviation',
                'hue_id','hue_fid','hue_crop_id','hue_crop_name','hue_SHAPE_AREA','hue_SHAPE_LEN','hue_point','hue_ts_complexity_cid_ce','hue_doy_of_maximum_dates','hue_doy_of_minimum_dates','hue_large_standard_deviation','hue_variance_larger_than_standard_deviation',
                ]  # List of columns to drop
df_259N_merge.drop(columns=cols_to_drop, inplace=True)

df_259N_merge.shape

#%%
df_259N_merge.head()

# %%
