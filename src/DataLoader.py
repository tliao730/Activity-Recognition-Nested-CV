import pandas as pd
import os
from pathlib import Path


class AReMDataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.activities = [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def load_all_data(self, test_size_map):
        """
        test_size_map: dictionary defining how many files to use as test set for each activity
        e.g. {"bending1": 2, "walking": 3, ...}
        """
        train_list = []
        test_list = []

        for activity in self.activities:
            dir_path = self.base_path / activity
            # get all csv files and sort them by the numeric part of the filename
            files = sorted([f for f in os.listdir(dir_path) if f.endswith('.csv')], 
                           key=lambda x: int(x.replace('dataset', '').replace('.csv', '')))
            
            test_cutoff = test_size_map.get(activity, 3) # take from test_size_map or default to 3

            for i, filename in enumerate(files, 1):
                file_path = dir_path / filename
                
                # skip 4 rows of explanation text at the beginning of each CSV file
                # adding engine='python' to handle potential encoding issues
                try:
                    df = pd.read_csv(file_path, skiprows=4, on_bad_lines='skip')
                    
                    # add activity label and source file index
                    df['activity'] = activity
                    df['sequence_id'] = i
                    
                    if i <= test_cutoff:
                        test_list.append(df)
                    else:
                        train_list.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        # high efficiency by concatenating all at once instead of in the loop
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        return train_df, test_df

# Example usage:
# loader = AReMDataLoader("../data/AReM")
# test_map = {"bending1": 2, "bending2": 2}
# train_data, test_data = loader.load_all_data(test_map)


if __name__ == "__main__":
    base_path = "../data/AReM"
    if Path(base_path).exists():
        loader = AReMDataLoader(base_path)
        test_map = {"bending1": 2, "bending2": 2}  # example test size map
        train_data, test_data = loader.load_all_data(test_map)
        print("Train Data Shape:", train_data.shape)
        print("Test Data Shape:", test_data.shape)