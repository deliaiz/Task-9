import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
# 1. 数据加载函数
def load_and_preprocess(train_path, test_path):
    # 读取 Excel 文件
    train_df = pd.read_excel(train_path, header=None)
    test_df = pd.read_excel(test_path, header=None)

    # 分离数值特征和类别特征
    X_train_numeric = train_df.iloc[:, :30]  # 前30列数值特征
    X_train_category = train_df.iloc[:, 30:31]  # 第31列类别特征
    y_train = train_df.iloc[:, 31]
    
    X_test_numeric = test_df.iloc[:, :30]
    X_test_category = test_df.iloc[:, 30:31]
    y_test = test_df.iloc[:, 31]

    # 2. 标准化数值特征
    scaler = StandardScaler()
    X_train_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_numeric),
        index=X_train_numeric.index,
        columns=X_train_numeric.columns
    )
    X_test_numeric_scaled = pd.DataFrame(
        scaler.transform(X_test_numeric),
        index=X_test_numeric.index,
        columns=X_test_numeric.columns
    )

    # 3. 独热编码处理第31列 (索引为30)
    X_train_cat = pd.get_dummies(X_train_category, columns=[30])
    X_test_cat = pd.get_dummies(X_test_category, columns=[30])
    
    # 对齐训练集和测试集的类别特征列
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    # 4. 合并标准化后的数值特征和独热编码后的类别特征
    X_train = pd.concat([X_train_numeric_scaled, X_train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_numeric_scaled, X_test_cat.reset_index(drop=True)], axis=1)

    # 将列名统一转为字符串
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    return X_train, y_train, X_test, y_test

try:
    train_file = '/public/home/liufj/perl5/finalwork_project/task9/dataset/train_data.xlsx'
    test_file = '/public/home/liufj/perl5/finalwork_project/task9/dataset/test_data.xlsx'

    X_train, y_train, X_test, y_test = load_and_preprocess(train_file, test_file)

    # 4. 集成学习模型
    rf = RandomForestRegressor(random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)
    ensemble = VotingRegressor([('rf', rf), ('gbr', gbr)])

    # 5. 使用交叉验证调参
    param_grid = {
        'rf__n_estimators': [50, 100, 150, 200],
        'rf__max_depth': [5, 10, 15, 20],
        'gbr__learning_rate': [ 0.01, 0.05, 0.08],
        'gbr__n_estimators': [50, 100, 150]
    }

    print("正在进行5折交叉验证调参（GridSearch）...")
    grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 获取最优模型和参数
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {best_params}\n")

    # 使用 KFold 保持与 GridSearchCV 相同的划分逻辑，这里使用KFlod的只是为了显示每一折的详细指标
    print("-" * 60)
    print(f"{'Fold':^6} | {'MSE':^10} | {'Mean Err':^10} | {'Var Err':^10} | {'RSE':^10}")
    print("-" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mses, fold_means, fold_vars, fold_rses = [], [], [], []

    # 使用 KFold 展示最优参数在不同子集上的稳定性
    for i, (t_idx, v_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr_fold, X_val_fold = X_train.iloc[t_idx], X_train.iloc[v_idx]
        y_tr_fold, y_val_fold = y_train.iloc[t_idx], y_train.iloc[v_idx]

        # 使用最优参数重新训练模型
        best_model.fit(X_tr_fold, y_tr_fold)
        y_val_pred = best_model.predict(X_val_fold)
        # 计算该折指标
        err = y_val_fold - y_val_pred
        mse = np.mean(err ** 2)
        m_err = np.mean(err)
        v_err = np.var(err)

        # 计算该折 RSE
        ss_res_fold = np.sum(err ** 2)
        ss_tot_fold = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
        rse_fold = ss_res_fold / ss_tot_fold

        # 存储结果
        fold_mses.append(mse)
        fold_means.append(m_err)
        fold_vars.append(v_err)
        fold_rses.append(rse_fold)

        print(f"{i + 1:^6} | {mse:10.4f} | {m_err:10.4f} | {v_err:10.4f} | {rse_fold:10.4f}")

    print("-" * 60)
    print(
        f"{'Average':^6} | {np.mean(fold_mses):10.4f} | {np.mean(fold_means):10.4f} | {np.mean(fold_vars):10.4f} | {np.mean(fold_rses):10.4f}")
    print("-" * 60 + "\n")

    # 6. 在独立测试集上进行最终预测
    # 重新在全体训练集上训练一次最优模型
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # 7. 报告最终评估指标
    errors = y_test - y_pred
    mean_err = np.mean(errors)
    var_err = np.var(errors)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    rse = ss_res / ss_tot

    print("=" * 30)
    print("      最终测试集评估报告")
    print("=" * 30)
    print(f"1. 误差均值 (Mean Error):      {mean_err:.4f}")
    print(f"2. 误差方差 (Variance of Error): {var_err:.4f}")
    print(f"3. 平方和相对误差 (RSE):         {rse:.4f}")
    print("=" * 30)

except Exception as e:
    print(f"运行过程中出现错误: {e}")