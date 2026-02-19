import pandas as pd
from feature_extraction import FeatExtrac
from model_evaluation import ModelEvaluator

def run_integration_test():
    print("🧪 開始整合測試...")
    
    # 1. 設定資料路徑 (請確保這是你電腦上的實際路徑)
    DATA_PATH = "../data/AReM"
    
    try:
        # 2. 測試特徵提取模組
        print("\n[Step 1/3] 初始化特徵提取器...")
        fe = FeatExtrac(DATA_PATH)
        
        # 3. 測試模型評估模組
        print("[Step 2/3] 初始化模型評估器...")
        evaluator = ModelEvaluator(fe)
        
        # 4. 執行小規模實驗 (只跑 l=1 和 l=2，節省時間)
        print("[Step 3/3] 執行 Nest CV 實驗 (l=1, 2)...")
        # 我們把 n_splits 設為 3，加速測試過程
        test_report = evaluator.run_l_segment_experiment(max_l=2, n_splits_outer=3)
        
        # 5. 檢查結果
        print("\n" + "✅" * 10)
        print("整合測試成功！")
        print("以下是測試結果摘要：")
        print(test_report)
        
        # 驗證欄位是否正確
        expected_cols = ['Split (l)', 'Num of Features (p)', 'Accuracy (Outer CV)', 'Accuracy (Inner RFECV)']
        if all(col in test_report.columns for col in expected_cols):
            print("\n📊 數據格式檢查：通過")
        
    except FileNotFoundError:
        print("\n❌ 測試失敗：找不到資料資料夾。請檢查 DATA_PATH 是否正確。")
    except Exception as e:
        print(f"\n❌ 測試中發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_integration_test()