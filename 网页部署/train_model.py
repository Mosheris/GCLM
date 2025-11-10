import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import json

print("=" * 60)
print("开始训练 XGBoost 模型...")
print("=" * 60)

# 1. 加载数据
print("\n[1/5] 加载数据...")
file_path = r"D:\R项目\胃癌肝转移\胃癌肝转移\Train.xlsx"  # 确保Train.xlsx在同一文件夹
try:
    data = pd.read_excel(file_path)
    print(f"✓ 数据加载成功！共 {len(data)} 条记录")
except FileNotFoundError:
    print(f"✗ 错误：找不到文件 {file_path}")
    print("请确保 Train.xlsx 与此脚本在同一文件夹")
    exit()

# 2. 准备特征和目标变量
print("\n[2/5] 准备特征...")

# 确保特征列存在
features = ["Gender", "Tumor.size", "Radiation", "Surgery",
            "Bone.metastasis", "Lung.metastasis", "N.stage"]

# 如果缺失某些特征列，提示错误
for feature in features:
    if feature not in data.columns:
        print(f"✗ 错误：数据中缺少特征 {feature}")
        exit()

X = data[features]
y = data['Liver.metastasis']

# 数据预处理：处理缺失值
X = X.fillna(X.mean())  # 使用均值填充缺失值

# 转换为数值型（确保兼容性）
y = y.astype(int)

print(f"✓ 特征数量: {len(features)}")
print(f"✓ 样本数量: {len(X)}")
print(f"✓ 阳性样本: {y.sum()} ({y.mean()*100:.1f}%)")

# 3. 训练XGBoost模型（使用R代码中的超参数）
print("\n[3/5] 训练 XGBoost 模型...")
print("超参数设置：")
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'eta': 0.1,  # 学习率
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'seed': 42
}
for k, v in params.items():
    print(f"  - {k}: {v}")

# 创建DMatrix
dtrain = xgb.DMatrix(X, label=y, feature_names=features)

# 训练模型
num_rounds = 300

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    verbose_eval=50  # 每50轮打印一次
)

# 4. 评估模型（使用训练集进行评估）
print("\n[4/5] 评估模型性能...")

y_train_pred = model.predict(dtrain)
train_auc = roc_auc_score(y, y_train_pred)

print(f"\n✓ 训练集 AUC: {train_auc:.4f}")

# 5. 保存模型
print("\n[5/5] 保存模型...")
model.save_model('xgb_liver_metastasis.json')
print("✓ 模型已保存为: xgb_liver_metastasis.json")

# 保存特征名称（供Streamlit使用）
with open('feature_names.json', 'w') as f:
    json.dump(features, f)
print("✓ 特征名称已保存为: feature_names.json")

print("\n" + "=" * 60)
print("🎉 模型训练完成！")
print("=" * 60)
print("\n下一步：运行 streamlit run app.py 启动网页应用")
