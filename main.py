import pandas as pd
from src.feature_extraction import FeatExtrac
from src.model_evaluation import ModelEvaluator
from pathlib import Path

def run_full_experiment():
    print("activating AReMactivity recognition...")
    print("-" * 50)

    # 1. initialize feature extraction and model evaluator
    DATA_PATH = "./data/AReM"
    fe = FeatExtrac(DATA_PATH)
    evaluator = ModelEvaluator(fe)

    # 2. execute the full experiment: test l = 1 to 20
    # attention: this may take several minutes since each l runs a 5-fold Nested CV
    max_segments = 20
    results_df = evaluator.run_l_segment_experiment(max_l=max_segments, n_splits_outer=5)

    # 3. output the complete report
    print("Logistic Regression + RFECV:")
    print(results_df.to_string(index=False))

    # 4. find the best configuration based on the highest outer CV accuracy
    best_idx = results_df['Accuracy (Outer CV)'].idxmax()
    best_row = results_df.loc[best_idx]

    print("\n" + "*" * 50)
    print("the best configuration based on highest outer CV accuracy:")
    print(f"the best window segmentation (l*): {int(best_row['Split (l)'])}")
    print(f"average number of features (p*): {int(best_row['Num of Features (p)'])}")
    print(f"the highest test accuracy: {best_row['Accuracy (Outer CV)']:.4f}")
    print("*" * 50)

    # 5. save the complete results to a CSV file for further analysis
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "experiment_results_l1_to_l20.csv"
    results_df.to_csv(file_path, index=False)   
    print(f"The complete results are saved to {file_path}")

if __name__ == "__main__":
    run_full_experiment()