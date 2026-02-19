from sklearn.pipeline import Pipeline

def nested_logistic_regression_eval(df_full_train, n_splits, solver, max_iter):
    X_full = df_full_train.drop(columns='label').copy()
    # 確保 y 是類別型或數值型標籤
    y_full = df_full_train['label'].copy()
    
    outer_validator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    outer_test_scores = []
    optimal_features_per_fold = []
    inner_rfecv_scores = [] 
    optimal_features_sets = []

    for train_index, test_index in outer_validator.split(X_full, y_full):
        X_train_raw, X_test_raw = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

        # --- 核心修正：正確的標準化流程 ---
        scaler = StandardScaler()
        # 1. 僅從訓練集學習平均值與標準差
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
        # 2. 套用到測試集 (不可重新 fit)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

        # Inner CV (RFECV)
        inner_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
        model = LogisticRegression(solver=solver, max_iter=max_iter)
        
        # 使用 RFECV 尋找最佳特徵子集
        # 建議加上 n_jobs=-1 加速運算
        selector = RFECV(estimator=model, cv=inner_validator, scoring='accuracy', n_jobs=-1)
        selector.fit(X_train, y_train)
        
        optimal_features = X_train.columns[selector.support_]
        optimal_features_sets.append(optimal_features.tolist())
        
        # 使用最佳特徵重新訓練模型並在 Outer Test 評估
        final_model = LogisticRegression(solver=solver, max_iter=max_iter)
        final_model.fit(X_train[optimal_features], y_train)
        
        y_pred = final_model.predict(X_test[optimal_features])
        outer_score = accuracy_score(y_test, y_pred)
        
        outer_test_scores.append(outer_score)
        optimal_features_per_fold.append(selector.n_features_)
        inner_rfecv_scores.append(selector.cv_results_['mean_test_score'].max())

    return np.mean(outer_test_scores), np.mean(optimal_features_per_fold), np.mean(inner_rfecv_scores), optimal_features_sets