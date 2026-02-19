import pandas as pd
import numpy as np
from pathlib import Path
import re

class FeatExtrac:
    def __init__(self, base_path_str="./data/AReM"):
        self.base_path = Path(base_path_str)
        
        self.dir_list = sorted([d for d in self.base_path.iterdir() if d.is_dir()])
        self.dir_map = {d.name: d for d in self.dir_list}

    def _natural_sort_key(self, s):
        """ using this to make sure dataset1, dataset2, ..., dataset10 are sorted in natural order """
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

    def get_features(self, data_type='train', l=1):
        """
        split the time series into l segments, and extract mean, median, std for each segment.
        """
        feature_names = ["mean", "median", "std"]
        all_data_list = []
        
        # dynamically generate feature column names based on the number of sensors (6), segments (l), and features (3)
        feature_columns = [
            f"{feat}{sensor}_seg{seg}"
            for sensor in range(1, 7)
            for seg in range(1, l + 1)
            for feat in feature_names
        ]

        for activity, path in self.dir_map.items():
            # acquire all CSV files and sort them in natural order (dataset1, dataset2, ..., dataset10)
            all_files = sorted([f for f in path.glob("*.csv")], key=self._natural_sort_key)
            
            # according to AReM convention, we take the first 2 files for bending activities and the first 3 files for other activities as test set
            cutoff = 2 if "bending" in activity else 3
            
            if data_type == 'train':
                files_to_process = all_files[cutoff:]
            else:
                files_to_process = all_files[:cutoff]
            
            # setting 1 for bending activities and 0 for others
            label = 'bending' if 'bending' in activity else 'other'

            for file_path in files_to_process:
                try:
                    # reading data and dropping the first column (time/index)
                    data = pd.read_csv(file_path, skiprows=5, sep=r'[,\s]+', header=None, engine='python')
                    data = data.drop(columns=[0]) 
                    
                    instance_features = []
                    
                    # iterating through each sensor column (assuming 6 sensors) and extracting features for each segment
                    for col_idx in data.columns:
                        time_series = data[col_idx]
                        
                        # split the time series into l segments (if l=1, it's just one segment which is the whole series)
                        segments = np.array_split(time_series, l)
                        
                        for seg in segments:
                            # extracting mean, median, std for the current segment and appending to instance features
                            instance_features.extend([seg.mean(), seg.median(), seg.std()])
                    
                    instance_features.append(label)
                    all_data_list.append(instance_features)
                    
                except Exception as e:
                    print(f"occuring error when reading {file_path.name} : {e}")

        # building the final DataFrame with dynamically generated feature columns and a label column
        final_df = pd.DataFrame(all_data_list, columns=feature_columns + ['label'])
        return final_df

# --- testing and executing  ---
if __name__ == "__main__":
    extractor = FeatExtrac("./data/AReM")
    
    print("extracting features for training set with L=1...")
    train_df_l1 = extractor.get_features(data_type='train', l=1)
    print(f"completed! training set shape: {train_df_l1.shape}")
    
    print("extracting features for test set with L=3...")
    test_df_l3 = extractor.get_features(data_type='test', l=3)
    print(f"completed! test set shape: {test_df_l3.shape}")

    # 4. results preview
    print(train_df_l1.columns[:15].tolist(), "...")
    print(train_df_l1.head())
    
    print(test_df_l3.columns[:15].tolist(), "...")
    print(test_df_l3.columns[:15].tolist(), "...")