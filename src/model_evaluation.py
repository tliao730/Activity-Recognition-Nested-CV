import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class ModelEvaluator:
    def __init__(self, feature_extractor):
        """
        Args:
            feature_extractor: 你的 FeatExtrac 實例，用來獲取資料。
        """
        self.fe = feature_extractor

    def nested_logistic_regression_eval(self, df_full_train, n_splits=5, solver='liblinear', max_iter=500):
        """
        使用巢狀交叉驗證來公平評估 RFECV 選出的特徵集。
        """
        X_full = df_full_train.drop(columns='label').copy()
        y_full = df_full_train['label'].copy()
        
        outer_validator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
        
        outer_test_scores = []
        optimal_features_per_fold = []
        inner_rfecv_scores = [] 
        all_optimal_features = []

        # Outer CV loop
        for train_index, test_index in outer_validator.split(X_full, y_full):
            X_train_raw, X_test_raw = X_full.iloc[train_index], X_full.iloc[test_index]
            y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

            # --- 關鍵修正：標準化 (Standardization) ---
            # 必須在每一折內獨立進行，且測試集只能用訓練集的參數進行 transform
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
            X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

            # Inner CV: 用於選擇特徵
            inner_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
            model = LogisticRegression(solver=solver, max_iter=max_iter)
            
            # 使用 RFECV 自動選擇最佳特徵數量
            selector = RFECV(estimator=model, cv=inner_validator, scoring='accuracy', n_jobs=-1)
            selector.fit(X_train, y_train)
            
            # 記錄這一折的最佳特徵
            optimal_features = X_train.columns[selector.support_]
            all_optimal_features.append(optimal_features.tolist())
            
            # 使用選出的特徵在 Outer Test 上進行公平評估
            final_model = LogisticRegression(solver=solver, max_iter=max_iter)
            final_model.fit(X_train[optimal_features], y_train)
            
            y_pred = final_model.predict(X_test[optimal_features])
            outer_test_scores.append(accuracy_score(y_test, y_pred))
            optimal_features_per_fold.append(selector.n_features_)
            inner_rfecv_scores.append(selector.cv_results_['mean_test_score'].max())

        return {
            'mean_outer_acc': np.mean(outer_test_scores),
            'mean_opt_p': np.mean(optimal_features_per_fold),
            'mean_inner_acc': np.mean(inner_rfecv_scores),
            'features_sets': all_optimal_features
        }

    def run_l_segment_experiment(self, max_l=20, n_splits_outer=5):
        """
        遍歷不同的時窗分割值 (l)，尋找最佳組合。
        """
        results = []

        for seg in range(1, max_l + 1):
            print(f"🔄 正在處理 l = {seg}...")
            # 從你的特徵提取模組獲取資料
            df_train = self.fe.get_features(data_type='train', l=seg)

            res = self.nested_logistic_regression_eval(
                df_train, n_splits=n_splits_outer
            )
            
            results.append({
                'Split (l)': seg,
                'Num of Features (p)': round(res['mean_opt_p']),
                'Accuracy (Outer CV)': round(res['mean_outer_acc'], 4),
                'Accuracy (Inner RFECV)': round(res['mean_inner_acc'], 4)
            })

        results_df = pd.DataFrame(results)
        return results_df

# --- 執行範例 ---
if __name__ == "__main__":
    from feature_extraction import FeatExtrac # 假設你的提取模組檔名
    
    # 1. 初始化
    fe = FeatExtrac("../data/AReM")
    evaluator = ModelEvaluator(fe)
    
    # 2. 執行實驗 (為了示範先跑 l=1~5)
    report_df = evaluator.run_l_segment_experiment(max_l=5)
    
    # 3. 輸出結果
    print("\n" + "="*50)
    print("📋 邏輯迴歸實驗報告")
    print("="*50)
    print(report_df)
    
    # 4. 找出最佳組合
    best_idx = report_df['Accuracy (Outer CV)'].idxmax()
    best = report_df.loc[best_idx]
    print(f"\n最佳 (l*, p*) 組合:")
    print(f"時窗分割 (l) = {int(best['Split (l)'])}")
    print(f"特徵數量 (p) ≈ {int(best['Num of Features (p)'])}")
    print(f"測試準確率 = {best['Accuracy (Outer CV)']}")