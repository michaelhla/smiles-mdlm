import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import datasets

def check_training_data_for_errors(training_data_file_path):
    try:
        dataset = datasets.load_from_disk(training_data_file_path)
        df = dataset.to_pandas()
        
        # Check for NaNs in the dataframe
        if df.isnull().values.any():
            print("NaN values found in the training data.")
            print(df[df.isnull().any(axis=1)])
        else:
            print("No NaN values found in the training data.")
        
        # Check for any other potential issues
        print("Data types of the columns:")
        print(df.dtypes)
        
        # print("Summary statistics of the data:")
        # print(df.describe())
        
    except Exception as e:
        print(f"An error occurred while reading the training data: {e}")

# Path to the training data file
training_data_file_path = "/root/smiles-mdlm/cache/chebi/train"
check_training_data_for_errors(training_data_file_path)
