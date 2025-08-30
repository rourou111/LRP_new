import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# 从 scikit-learn 中导入我们需要的模块
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import yaml
from sklearn.linear_model import LogisticRegression

# --- 加载配置文件 ---
# 因为我们的脚本在 v2_stacked_model 文件夹内，所以需要用 ../ 来访问上一层的config.yaml
with open("../config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("所有机器学习库都已成功导入！")
print("两阶段分类器训练脚本已准备就绪。")

# =============================================================================
# 步骤一：数据加载与通用预处理
# =============================================================================

# --- 1. 自动寻找最新的指纹数据文件 ---
# 从config文件中获取路径
runs_dir = config['output_paths']['runs_directory']
list_of_run_dirs = glob.glob(os.path.join(runs_dir, '*/'))
if not list_of_run_dirs:
    print("\n错误：在 'runs' 文件夹下找不到任何运行记录。")
    exit()

latest_run_dir = max(list_of_run_dirs, key=os.path.getctime)
fingerprint_file_path = os.path.join(latest_run_dir, 'vulnerability_fingerprints.csv')

print(f"\n正在从最新的运行记录中加载数据: {fingerprint_file_path}")

try:
    data = pd.read_csv(fingerprint_file_path)
    print(f"成功加载 {len(data)} 个样本。")
except FileNotFoundError:
    print(f"\n错误：在路径 '{fingerprint_file_path}' 中找不到 vulnerability_fingerprints.csv 文件。")
    exit()

# --- 2. 分离特征 (X) 与原始标签 (y) ---
X = data.drop('vulnerability_type', axis=1)
y_str = data['vulnerability_type']

# --- 3. 标签编码 ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_str)
label_mapping = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
print("\n标签已成功编码为数字:")
print(label_mapping)

# --- 4. 划分总数据集 (为最终评估做准备) ---
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 5. 全局预处理：处理无穷大值 ---
# <<< 关键修正点 1 >>>
# 在数据分割后，立即对训练集和测试集进行全局的无穷大值处理。
# 这样可以确保后续所有基于这两个数据集的操作都是安全的。
print("\n正在对训练集和测试集进行无穷大值预处理...")
X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)
print("预处理完成。")


# =============================================================================
# 阶段一：训练模型一 (“漂移”识别器 - Generalist)
# =============================================================================
print("\n" + "="*50)
print("阶段一：开始训练模型一 ('漂移'识别器)")
print("="*50)

# --- 1.1 从总训练集中创建模型一的二元标签 ---
drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
y_train1 = np.where(y_train_full == drift_label_encoded, 1, 0)

# --- 1.2 数据修复与标准化 ---
# <<< 关键修正点 2 >>>
# 由于 X_train_full 已经被处理过，我们直接使用它即可。
# 原先对 X_train1 的替换操作可以安全地移除。
X_train1 = X_train_full.copy() 
# X_train1.replace([np.inf, -np.inf], np.nan, inplace=True) # <-- 这行已不再需要，可以删除

imputer1 = SimpleImputer(strategy='median')
X_train1_imputed = imputer1.fit_transform(X_train1)
scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1_imputed)



# --- 1.3 训练模型一 ---
model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model1.fit(X_train1_scaled, y_train1)
print("模型一训练完成。")
# --- 新增：快速验证模型一在训练集上的性能 ---
from sklearn.metrics import classification_report
pred1_train = model1.predict(X_train1_scaled)
print("\n--- 模型一 (分诊台) 在训练集上的性能报告 ---")
print(classification_report(y_train1, pred1_train, target_names=['非漂移 (0)', '漂移 (1)']))
# --- 验证结束 ---
# =============================================================================
# 阶段二：构建“专家组决策系统” (Specialist System)
# =============================================================================
print("\n" + "="*50)
print("阶段二：开始构建'专家组决策系统'")
print("="*50)

# --- 2.1 从总训练集中筛选出需要专家会诊的数据 ---
non_drift_train_mask = (y_train1 == 0) # 找到训练集中所有“非漂移”的样本
X_train_experts = X_train_full[non_drift_train_mask]
y_train_experts_raw = y_train_full[non_drift_train_mask]

# 为专家组的数据创建新的标签 (对抗 vs. 噪声)
label_encoder2 = LabelEncoder()
y_train_experts = label_encoder2.fit_transform(y_train_experts_raw)

print(f"已筛选出 {len(X_train_experts)} 个'对抗 vs. 噪声'样本，交由专家组处理。")
# --- 2.2 定义每个专家的特征领域 (升级版) ---

# 原有的动态特征专家，现在得到了新武器“动态小波能量比变化率”
features_dynamic = [
    'wasserstein_dist', 'cosine_similarity', 'kl_divergence_pos', 
    'kl_divergence_neg', 'std_dev_diff', 'kurtosis_diff',
    'dynamic_wavelet_ratio_change' # <--- 新增
]

# 频域分析专家只关注傅里叶变换的高频特征
features_frequency = ['high_freq_ratio']

# 纹理学专家，现在得到了新武器“低频子带结构失真度”
features_texture = [
    'contrast', 'homogeneity', 'energy', 'correlation',
    'll_distortion' # <--- 新增
]

# 全新专家：“高敏哨兵”，它的眼中只有Z-score这一个最敏感的指标
features_sensitivity = ['ratio_zscore'] # <--- 新增
# --- 2.3 训练专家并使用交叉验证生成元特征 (升级版) ---
# ... (准备数据的代码 X_train_experts_scaled_df 不变) ...
# 使用阶段一的预处理器处理专家数据
X_train_experts_imputed = imputer1.transform(X_train_experts)
X_train_experts_scaled = scaler1.transform(X_train_experts_imputed)
X_train_experts_scaled_df = pd.DataFrame(X_train_experts_scaled, 
                                        columns=X_train_experts.columns, 
                                        index=X_train_experts.index)

# 创建专家模型实例 (增加一位新专家)
dynamic_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
frequency_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
texture_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
sensitivity_expert = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # <--- 新增

# 获取每个专家的预测概率 (增加一位新专家)
dynamic_opinions = cross_val_predict(dynamic_expert, X_train_experts_scaled_df[features_dynamic], y_train_experts, cv=5, method='predict_proba')
frequency_opinions = cross_val_predict(frequency_expert, X_train_experts_scaled_df[features_frequency], y_train_experts, cv=5, method='predict_proba')
texture_opinions = cross_val_predict(texture_expert, X_train_experts_scaled_df[features_texture], y_train_experts, cv=5, method='predict_proba')
sensitivity_opinions = cross_val_predict(sensitivity_expert, X_train_experts_scaled_df[features_sensitivity], y_train_experts, cv=5, method='predict_proba') # <--- 新增

# 将所有专家的意见合并，形成新的特征集 (现在有4位专家的意见)
X_train_meta = np.hstack([dynamic_opinions, frequency_opinions, texture_opinions, sensitivity_opinions]) # <--- 修改
print("专家会诊完成，已形成元特征集。")


# --- 2.4 训练最终决策者 (Meta-Classifier) (升级版) ---
# ... (为每个专家模型进行真实的fit，增加一位新专家) ...
print("\n各位专家正在学习总结...")
dynamic_expert.fit(X_train_experts_scaled_df[features_dynamic], y_train_experts)
frequency_expert.fit(X_train_experts_scaled_df[features_frequency], y_train_experts)
texture_expert.fit(X_train_experts_scaled_df[features_texture], y_train_experts)
sensitivity_expert.fit(X_train_experts_scaled_df[features_sensitivity], y_train_experts) # <--- 新增
print("专家学习完成。")

# ... (训练最终决策者的代码不变，它会自动适应新的输入) ...
meta_classifier = LogisticRegression(random_state=42)
meta_classifier.fit(X_train_meta, y_train_experts)
print("最终决策者训练完成！")
# =============================================================================
# 阶段三：对完整的“专家组决策系统”进行最终评估
# =============================================================================
print("\n" + "="*50)
print("阶段三：开始对两阶段系统进行最终评估")
print("="*50)

# --- 3.1 准备测试集数据 ---
# 注意：我们必须使用在训练集上fit过的imputer和scaler来transform测试集
X_test_imputed = imputer1.transform(X_test_full)
X_test_scaled = scaler1.transform(X_test_imputed)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_full.columns, index=X_test_full.index)

# --- 3.2 执行两阶段预测流程 ---
print("测试样本进入系统...")
# a. 首先，所有样本都由模型一“分诊台”进行初步诊断
print(" -> 步骤1: '分诊台' (模型一) 正在进行初步诊断...")
pred1_test = model1.predict(X_test_scaled)

# b. 初始化一个数组，用于存放我们最终的预测结果
final_predictions = np.zeros_like(y_test_full)

# c. 找到被模型一判断为“非漂移”，需要专家会诊的样本
non_drift_test_mask = (pred1_test == 0)
X_test_for_experts = X_test_scaled_df[non_drift_test_mask]
print(f" -> '分诊台'诊断完毕: {len(X_test_for_experts)} 个样本被提交至'专家组'。")

# --- 这是新的“专家组”预测流程 (升级版) ---
if len(X_test_for_experts) > 0:
    print(" -> 步骤2: '专家组'开始对疑难样本进行会诊...")
    # d. 各位专家对需要会诊的样本，分别给出自己的意见（预测概率）
    dynamic_opinions_test = dynamic_expert.predict_proba(X_test_for_experts[features_dynamic])
    frequency_opinions_test = frequency_expert.predict_proba(X_test_for_experts[features_frequency])
    texture_opinions_test = texture_expert.predict_proba(X_test_for_experts[features_texture])
    sensitivity_opinions_test = sensitivity_expert.predict_proba(X_test_for_experts[features_sensitivity]) # <--- 新增

    # e. 将所有专家的意见合并，形成元特征
    X_test_meta = np.hstack([dynamic_opinions_test, frequency_opinions_test, texture_opinions_test, sensitivity_opinions_test]) # <--- 修改

    
    # f. “最终决策者”根据所有专家的意见，做出最终裁决
    expert_predictions = meta_classifier.predict(X_test_meta)
    
    # g. 将专家组的预测结果(0, 1)转换回原始的三分类标签
    expert_predictions_original_labels = label_encoder2.inverse_transform(expert_predictions)
    
    # h. 将专家组的诊断结果，填入我们最终的预测报告中
    final_predictions[non_drift_test_mask] = expert_predictions_original_labels
    print(" -> '专家组'会诊完毕。")

# i. 将模型一直接诊断为“漂移”的结果，也填入最终报告
drift_label_encoded = list(label_encoder.classes_).index('drift_parameter')
final_predictions[pred1_test == 1] = drift_label_encoded
print("...所有样本预测流程结束。")


# --- 3.3 评估并展示最终系统性能 ---
print("\n--- 两阶段专家组决策系统最终性能评估 ---")
print(f"最终准确率: {accuracy_score(y_test_full, final_predictions):.4f}")

print("\n最终分类报告:")
print(classification_report(y_test_full, final_predictions, target_names=label_encoder.classes_))

print("\n最终混淆矩阵:")
final_cm = confusion_matrix(y_test_full, final_predictions)

# --- 3.4 可视化最终混淆矩阵 ---
plt.figure(figsize=(10, 8))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Final Confusion Matrix for the Two-Stage Expert System')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# 保存图像到最新的运行文件夹
if 'latest_run_dir' in locals() and os.path.exists(latest_run_dir):
    plt.savefig(os.path.join(latest_run_dir, 'final_confusion_matrix_expert_system.png'))
    print(f"\n混淆矩阵图像已保存到: {latest_run_dir}")
plt.show()

print("\n脚本执行完毕！")