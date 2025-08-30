import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# 导入 Captum 库中的核心模块：LRP 和可视化工具
from captum.attr import LRP
from captum.attr import visualization as viz
import yaml

# --- 加载配置文件 ---
# 因为我们的脚本在 v2_stacked_model 文件夹内，所以需要用 ../ 来访问上一层的config.yaml
with open("../config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("所有库导入成功！")

# =============================================================================
# 第1部分：基础设置 (修正部分)
# --- 我们把 device 的定义提前到这里 ---
# =============================================================================
# 设置设备（如果电脑有NVIDIA显卡，则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")


# =============================================================================
# 第2部分：加载模型和数据
# =============================================================================
# --- 加载模型 ---
model = models.resnet18(weights=None)
model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
print(f"正在从本地路径加载预训练模型: {model_path}")

try:
    # --- 现在使用 device 变量时，它已经被定义了 ---
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
except FileNotFoundError:
    print(f"错误: 在路径 {model_path} 未找到权重文件。")
    exit()

model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)
model.eval()
print("\n模型已通过您的原始方法加载成功！")

# --- 加载漏洞数据 ---
# 我们需要从上一级目录读取这个文件
vulnerabilities_file_path = 'all_vulnerabilities.pkl'
print(f"\n正在从路径加载漏洞数据: {vulnerabilities_file_path}")

try:
    with open(vulnerabilities_file_path, 'rb') as f:
        all_vulnerabilities = pickle.load(f)
    print(f"成功加载 {len(all_vulnerabilities)} 个漏洞样本！")

    if all_vulnerabilities:
        vulnerability_sample = all_vulnerabilities[0]
        print("\n已选中第一个漏洞样本进行处理。")
    else:
        print("错误：漏洞列表中没有样本。请先运行 generate_vulnerabilities.py")
        exit()
except FileNotFoundError:
    print("错误：找不到 all_vulnerabilities.pkl 文件。")
    exit()


# =============================================================================
# 第3部分 (新版): 成对计算 H_clean 和 H_vuln
# =============================================================================
# 1. 创建一个总的“档案馆”文件夹 'runs'
base_output_dir = 'runs'
os.makedirs(base_output_dir, exist_ok=True)

# 2. 创建一个唯一的、带时间戳的专属“档案室”文件夹
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_output_dir = os.path.join(base_output_dir, timestamp)
os.makedirs(run_output_dir, exist_ok=True)
print(f"\n本次运行的所有输出将被保存在: {run_output_dir}")

# 我们将创建一个新列表，用于存储成对的热力图数据
paired_heatmaps_data = []

print(f"\n--- 开始为 {len(all_vulnerabilities)} 个漏洞样本生成成对的热力图 ---")

# 初始化LRP分析器 (我们只需要初始化一次)
lrp = LRP(model)

# 使用 for 循环遍历每一个漏洞样本
for i, sample in enumerate(all_vulnerabilities):
    
    # --- 1. 准备计算 H_clean 和 H_vuln 所需的全部“原材料” ---
    
    # --- 这是修改后的部分 ---
    # 根据漏洞类型，使用正确的键名来获取原始干净图片
    if sample['vulnerability_type'] == 'drift_parameter':
        original_image = sample['image'].to(device)
    else:
        original_image = sample['original_image'].to(device)
    # -------------------------

    # 真实、正确的标签 (所有样本类型都有)
    true_label = int(sample['label'])

    # 根据漏洞类型，确定失效图片和错误预测的标签
    if sample['vulnerability_type'] == 'adversarial_pgd':
        vulnerable_image = sample['adversarial_image'].to(device)
        predicted_class = int(sample['adversarial_pred'])
    elif sample['vulnerability_type'] == 'noise_gaussian':
        vulnerable_image = sample['noisy_image'].to(device)
        predicted_class = int(sample['noisy_pred'])
    elif sample['vulnerability_type'] == 'drift_parameter':
        # 漂移漏洞的键名不同，我们用 original_image 作为失效图片
        vulnerable_image = sample['image'].to(device) 
        predicted_class = int(sample['drifted_pred'])
    else:
        print(f"警告: 跳过未知漏洞类型 at index {i}")
        continue

    print(f"  处理样本 {i+1}/{len(all_vulnerabilities)}: 真实类别: {true_label}, 模型错误预测为: {predicted_class}")

    # --- 2. 执行两次LRP计算 ---

    # 计算A (H_clean): 干净图片 -> 对应 -> 真实标签
    attribution_clean = lrp.attribute(original_image.unsqueeze(0), target=true_label)

    # 计算B (H_vuln): 失效图片 -> 对应 -> 错误标签
    attribution_vuln = lrp.attribute(vulnerable_image.unsqueeze(0), target=predicted_class)

    # --- 3. 将成对的结果存入一个字典 ---

    # 我们使用 .cpu() 来确保保存的是CPU上的Tensor，更容易迁移
    paired_data = {
        "h_clean": attribution_clean.squeeze(0).cpu(),
        "h_vuln": attribution_vuln.squeeze(0).cpu(),
        "vulnerability_type": sample['vulnerability_type'],
        "true_label": true_label,
        "predicted_class": predicted_class
    }

    # 将这个包含了一对热力图的字典，添加到我们的总列表中
    paired_heatmaps_data.append(paired_data)

print("\n--- 所有样本的成对热力图已计算完毕 ---")

# =============================================================================
# 第4部分 (新版)：保存包含成对热力图的最终结果
# =============================================================================
# 将输出文件名和我们新建的专属文件夹路径结合起来
output_filename = os.path.join(run_output_dir, 'paired_heatmaps.pkl')
if os.path.exists(output_filename):
    # 如果存在，则删除它
    os.remove(output_filename)
    print(f"\n已删除旧的配对热力图文件: '{output_filename}'")
with open(output_filename, 'wb') as f:
    pickle.dump(paired_heatmaps_data, f)

print(f"\n包含成对热力图的最终结果已成功保存到: {output_filename}")
print("您现在可以进入下一个阶段：指纹提取与分析！")