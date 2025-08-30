import pickle
import numpy as np
from numpy.fft import fft2, fftshift
import pandas as pd
import torch
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
import pywt
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
try:
    with open('all_vulnerabilities.pkl', 'rb') as f:
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

# 从 scipy.stats 中导入用于计算 推土机距离、KL散度 和 峰度 的函数
from scipy.stats import wasserstein_distance, entropy as kl_divergence, kurtosis

# 从 scipy.spatial.distance 中导入用于计算 余弦距离 的函数
from scipy.spatial.distance import cosine as cosine_distance

print("所有必要的库都已成功导入！")
print("指纹提取器已准备就绪。")
# =============================================================================
# 步骤二：自动寻找并加载最新的热力图数据
# =============================================================================

# a. 找到 'runs' 文件夹下所有带时间戳的子文件夹
# 我们需要从上一级目录 (../) 开始寻找
list_of_run_dirs = glob.glob('runs/*/')
if not list_of_run_dirs:
    print("\n错误：在 'runs' 文件夹下找不到任何运行记录。")
    print("请确保您已经成功运行了 'generate_heatmaps.py' 脚本。")
    exit()

# b. 根据创建时间，找到最新的那个文件夹
latest_run_dir = max(list_of_run_dirs, key=os.path.getctime)
heatmap_file_path = os.path.join(latest_run_dir, 'paired_heatmaps.pkl')

print(f"\n正在从最新的运行记录中加载数据: {heatmap_file_path}")

# c. 加载数据
try:
    with open(heatmap_file_path, 'rb') as f:
        paired_heatmaps = pickle.load(f)
    print(f"成功加载 {len(paired_heatmaps)} 组配对的热力图数据。")
except FileNotFoundError:
    print(f"\n错误：在路径 '{heatmap_file_path}' 中找不到 paired_heatmaps.pkl 文件。")
    exit()
# =============================================================================
# 步骤三：实现六个核心的特征计算函数
# =============================================================================

def calculate_wasserstein(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的推土机距离 (Wasserstein Distance)。

    Args:
        h1_tensor (torch.Tensor): 第一张热力图。
        h2_tensor (torch.Tensor): 第二张热力图。

    Returns:
        float: 两张热力图之间的推土机距离。
    """
    # 步骤 1: 将输入的PyTorch Tensor转换为NumPy数组
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    # 步骤 2: 将二维的热力图矩阵展平（flatten）为一维向量
    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # 步骤 3: 调用scipy函数计算并返回推土机距离
    distance = wasserstein_distance(h1_flat, h2_flat)
    
    return distance    

def calculate_cosine_similarity(h1_tensor, h2_tensor):
    """
    计算两张热力图之间的余弦相似度 (Cosine Similarity)。
    (更健壮的版本，增加了对零向量的检查)
    """
    h1_np = h1_tensor.detach().numpy()
    h2_np = h2_tensor.detach().numpy()

    h1_flat = h1_np.flatten()
    h2_flat = h2_np.flatten()

    # --- 安全检查 ---
    # 如果任一向量的模长（L2范数）接近于零，则它们是零向量
    if np.linalg.norm(h1_flat) < 1e-10 or np.linalg.norm(h2_flat) < 1e-10:
        # 两个零向量之间的相似度可以定义为1，一个零和一个非零向量相似度为0
        return 1.0 if np.linalg.norm(h1_flat) < 1e-10 and np.linalg.norm(h2_flat) < 1e-10 else 0.0
    # -----------------

    distance = cosine_distance(h1_flat, h2_flat)
    similarity = 1 - distance

    return similarity
def calculate_kl_divergences(h_clean_tensor, h_vuln_tensor):
    """
    计算干净热力图与失效热力图之间，正、负贡献分布的KL散度。
    (新版：采用更稳健的平滑方法，从根源上防止无穷大值)
    
    Args:
        h_clean_tensor (torch.Tensor): 干净样本的热力图 (基准分布 P)。
        h_vuln_tensor (torch.Tensor): 失效样本的热力图 (近似分布 Q)。

    Returns:
        tuple[float, float]: 返回一个元组，包含 (正贡献KL散度, 负贡献KL散度)。
    """
    epsilon = 1e-10  # 定义一个极小值，用于平滑

    def _get_smoothed_distributions(h_tensor):
        """内部辅助函数，用于生成平滑后的正、负子分布"""
        h_flat = h_tensor.detach().flatten()
        
        # 分离正、负贡献
        h_pos = torch.clamp(h_flat, min=0)
        h_neg = torch.abs(torch.clamp(h_flat, max=0))
        
        # --- 核心修改：先平滑，再归一化 ---
        # 1. 给所有像素点都加上一个极小的基础概率值 (平滑)
        p_pos_smooth = h_pos + epsilon
        p_neg_smooth = h_neg + epsilon
        
        # 2. 在平滑后的新分布上进行归一化
        p_pos_normalized = p_pos_smooth / torch.sum(p_pos_smooth)
        p_neg_normalized = p_neg_smooth / torch.sum(p_neg_smooth)
        
        return p_pos_normalized.numpy(), p_neg_normalized.numpy()

    # 步骤 1: 为两张热力图分别准备平滑、归一化后的正、负子概率分布
    p_clean_pos, p_clean_neg = _get_smoothed_distributions(h_clean_tensor)
    p_vuln_pos, p_vuln_neg = _get_smoothed_distributions(h_vuln_tensor)

    # 步骤 2: 分别计算正、负贡献的KL散度
    kl_pos = kl_divergence(p_clean_pos, p_vuln_pos)
    kl_neg = kl_divergence(p_clean_neg, p_vuln_neg)
    
    return kl_pos, kl_neg
def calculate_std_dev_diff(h_clean_tensor, h_vuln_tensor):
    """
    计算 H_vuln 和 H_clean 之间标准差的差值 (动态特征)。
    """
    # 分别计算两张热力图的标准差
    std_clean = h_clean_tensor.detach().numpy().std()
    std_vuln = h_vuln_tensor.detach().numpy().std()

    # 返回差值
    return std_vuln - std_clean
def calculate_kurtosis_diff(h_clean_tensor, h_vuln_tensor):
    """
    计算 H_vuln 和 H_clean 之间峰度的差值 (动态特征)。
    """
    # 分别计算两张热力图的峰度
    kurt_clean = kurtosis(h_clean_tensor.detach().numpy().flatten())
    kurt_vuln = kurtosis(h_vuln_tensor.detach().numpy().flatten())

    # 返回差值
    return kurt_vuln - kurt_clean
def calculate_high_freq_energy_ratio(h_vuln_tensor, high_freq_band=0.25):
    """
    计算单张热力图的高频能量占比。
    (修正版：增加了对三通道热力图的处理)

    Args:
        h_vuln_tensor (torch.Tensor): 失效样本的热力图。
        high_freq_band (float): 定义高频区域的阈值，例如0.25代表离中心25%以外的区域。

    Returns:
        float: 高频能量占总能量的比例。
    """
    # 步骤 1: 将Tensor转换为NumPy数组
    h_np_3channel = h_vuln_tensor.detach().numpy()
    
    # --- 这是新添加的步骤：将三通道热力图转换为单通道灰度图 ---
    # 我们通过在通道维度(axis=0)上取平均值来实现
    if h_np_3channel.ndim == 3 and h_np_3channel.shape[0] == 3:
        h_np = h_np_3channel.mean(axis=0)
    else:
        h_np = h_np_3channel # 如果已经是单通道，则直接使用
    # ----------------------------------------------------------

    # 步骤 2: 执行二维快速傅里叶变换
    f_transform = fft2(h_np)
    
    # 步骤 3: 将零频率分量移动到频谱中心
    f_transform_shifted = fftshift(f_transform)
    
    # 步骤 4: 计算能量谱（幅度的平方）
    magnitude_spectrum = np.abs(f_transform_shifted)**2
    
    # 步骤 5: 定义并计算高频区域的能量
    rows, cols = h_np.shape # <-- 现在这里的 h_np 已经是二维的了，不会再报错
    center_row, center_col = rows // 2, cols // 2
    
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = x**2 + y**2 > (min(center_row, center_col) * high_freq_band)**2
    
    high_freq_energy = np.sum(magnitude_spectrum[mask])
    total_energy = np.sum(magnitude_spectrum)
    
    # 步骤 6: 计算并返回高频能量占比
    if total_energy < 1e-10:
        return 0.0
        
    ratio = high_freq_energy / total_energy
    
    return ratio

def calculate_texture_features(h_vuln_tensor):
    """
    计算单张热力图的多种纹理特征。

    Args:
        h_vuln_tensor (torch.Tensor): 失效样本的热力图。

    Returns:
        dict: 一个包含多种纹理特征的字典。
    """
    # --- 步骤 1: 预处理热力图 ---
    
    # a. 将Tensor转换为NumPy数组
    h_np_3channel = h_vuln_tensor.detach().numpy()
    
    # b. 将三通道热力图转换为单通道灰度图 (通过取平均值)
    if h_np_3channel.ndim == 3 and h_np_3channel.shape[0] == 3:
        h_np_gray = h_np_3channel.mean(axis=0)
    else:
        h_np_gray = h_np_3channel

    # c. 将浮点数值归一化到 0-255 的整数范围
    # GLCM函数需要整数输入来代表不同的“灰度等级”
    # 我们先将数值范围缩放到 0-255
    h_min, h_max = h_np_gray.min(), h_np_gray.max()
    if h_max - h_min < 1e-10:
        # 如果图像是纯色的，则所有纹理特征都为0或1
        return {'contrast': 0.0, 'homogeneity': 1.0, 'energy': 1.0, 'correlation': 1.0}
        
    h_normalized = (h_np_gray - h_min) / (h_max - h_min) * 255.0
    # d. 转换为无符号8位整数类型 (uint8)
    h_int = h_normalized.astype(np.uint8)

# --- 步骤 2: 计算灰度共生矩阵 (GLCM) ---

    # a. 定义参数
    # distances: 我们考虑像素邻居的距离，这里只考虑紧邻的1个像素。
    # angles: 我们考虑4个方向（0度-水平，45度-斜对角，90度-垂直, 135度-反斜对角）。
    # levels: 灰度等级，我们之前归一化到了0-255，所以是256。
    # symmetric & normed: 标准参数，保持默认即可。
    glcm = graycomatrix(h_int, distances=[1], angles=[0, np.pi/4, np.pi/2, np.pi*3/4], levels=256, symmetric=True, normed=True)

    # --- 步骤 3: 从GLCM中提取四个纹理特征 ---
    
    # a. 对四个方向的结果取平均值，得到一个更稳健的、与方向无关的特征值
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # b. 将所有结果打包成一个字典返回
    texture_features = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }
    
    return texture_features
def calculate_dynamic_wavelet_ratio(h_clean_tensor, h_vuln_tensor):
    """特征一：计算动态小波能量比变化率"""
    
    def get_ratio(h_tensor):
        h_np = h_tensor.detach().numpy().mean(axis=0) # 转为灰度图
        coeffs = pywt.dwt2(h_np, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # 计算能量 (平方和)
        energy_ll = np.sum(LL**2)
        energy_high_freq = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
        
        # 加上一个极小值防止除以零
        return energy_high_freq / (energy_ll + 1e-10)

    ratio_clean = get_ratio(h_clean_tensor)
    ratio_vuln = get_ratio(h_vuln_tensor)
    
    change_ratio = (ratio_vuln - ratio_clean) / (ratio_clean + 1e-10)
    return change_ratio

def calculate_ll_distortion(h_clean_tensor, h_vuln_tensor):
    """特征二：计算低频子带结构失真度"""
    
    def get_ll_texture_vec(h_tensor):
        h_np = h_tensor.detach().numpy().mean(axis=0)
        coeffs = pywt.dwt2(h_np, 'haar')
        LL, _ = coeffs
        
        # 归一化到 0-255 的整数范围以计算GLCM
        ll_min, ll_max = LL.min(), LL.max()
        if ll_max - ll_min < 1e-10: return np.array([0.0, 1.0, 1.0, 1.0])
        ll_normalized = (LL - ll_min) / (ll_max - ll_min) * 255.0
        ll_int = ll_normalized.astype(np.uint8)
        
        glcm = graycomatrix(ll_int, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        return np.array([contrast, homogeneity, energy, correlation])

    vec_c = get_ll_texture_vec(h_clean_tensor)
    vec_v = get_ll_texture_vec(h_vuln_tensor)
    
    # 计算余弦距离: 1 - (a dot b) / (||a|| * ||b||)
    dist = 1 - np.dot(vec_c, vec_v) / (np.linalg.norm(vec_c) * np.linalg.norm(vec_v) + 1e-10)
    return dist

# =============================================================================
# 步骤四：批量处理所有样本，提取指纹
# =============================================================================
# --- 特征三的前置步骤：为Z-score校准基准 ---
print("\n为特征三(Z-score)校准噪声基准...")
noise_ratios = []
for data_pair in paired_heatmaps:
    if data_pair['vulnerability_type'] == 'noise_gaussian':
        h_clean = data_pair['h_clean']
        h_vuln = data_pair['h_vuln']
        # 复用上面函数中的逻辑来计算静态能量比
        def get_ratio(h_tensor):
            h_np = h_tensor.detach().numpy().mean(axis=0)
            coeffs = pywt.dwt2(h_np, 'haar')
            LL, (LH, HL, HH) = coeffs
            energy_ll = np.sum(LL**2)
            energy_high_freq = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
            return energy_high_freq / (energy_ll + 1e-10)
        # 我们用 H_vuln 来计算静态比值
        noise_ratios.append(get_ratio(h_vuln))

mu_noise = np.mean(noise_ratios)
sigma_noise = np.std(noise_ratios)
print(f"噪声基准校准完成: 平均值={mu_noise:.4f}, 标准差={sigma_noise:.4f}")

# 创建一个列表，用于存储每个漏洞样本的最终指纹数据
fingerprints_list = []

print(f"\n--- 开始为 {len(paired_heatmaps)} 组热力图提取指纹 ---")

# 遍历我们从 .pkl 文件中加载的每一组成对的热力图数据
for i, data_pair in enumerate(paired_heatmaps):
    
    # 从数据对中取出干净热力图、失效热力图和漏洞类型标签
    h_clean = data_pair['h_clean']
    h_vuln = data_pair['h_vuln']
    vuln_type = data_pair['vulnerability_type']
    
    print(f"\r  正在处理样本 {i+1}/{len(paired_heatmaps)}", end="")
    
    # --- 调用我们之前定义的所有函数，计算6个特征值 ---
    
    # 1. 对比性特征
    wasserstein = calculate_wasserstein(h_clean, h_vuln)
    cosine_sim = calculate_cosine_similarity(h_clean, h_vuln)
    kl_pos, kl_neg = calculate_kl_divergences(h_clean, h_vuln)
    
    # 2. 内在性特征
# 2. 内在动态特征 (描述变化过程)
    std_diff = calculate_std_dev_diff(h_clean, h_vuln)
    kurt_diff = calculate_kurtosis_diff(h_clean, h_vuln)
    high_freq_ratio = calculate_high_freq_energy_ratio(h_vuln)
    # 3. 新增的纹理特征
    texture_feats = calculate_texture_features(h_vuln)
    # 调用新特征计算函数
    dynamic_ratio_change = calculate_dynamic_wavelet_ratio(h_clean, h_vuln)
    ll_distortion = calculate_ll_distortion(h_clean, h_vuln)
    
    # 计算特征三 (Z-score)
    # 复用 get_ratio 逻辑来获取当前样本的静态比值
    static_ratio_sample = get_ratio(h_vuln) 
    ratio_zscore = np.abs(static_ratio_sample - mu_noise) / (sigma_noise + 1e-10)

    # --- 将所有结果存入一个字典 ---
# --- 将所有结果存入一个字典 ---
    fingerprint_data = {
        # 原有的对比性特征
        'wasserstein_dist': wasserstein,
        'cosine_similarity': cosine_sim,
        'kl_divergence_pos': kl_pos,
        'kl_divergence_neg': kl_neg,

        # 原有的内在动态特征
        'std_dev_diff': std_diff,
        'kurtosis_diff': kurt_diff,
        
        # 原有的内在静态特征
        'high_freq_ratio': high_freq_ratio,
        
        # 原有的纹理特征 (通过解包字典添加)
        **texture_feats,

        # --- 新增的三个核心特征 ---
        'dynamic_wavelet_ratio_change': dynamic_ratio_change,
        'll_distortion': ll_distortion,
        'ratio_zscore': ratio_zscore,
        
        # 漏洞类型标签 (只保留一个)
        'vulnerability_type': vuln_type
    }
    
    # 将这个样本的指纹字典添加到总列表中
    fingerprints_list.append(fingerprint_data)

print("\n--- 所有指纹已成功提取 ---")

# =============================================================================
# 步骤五：使用Pandas将结果保存为CSV文件
# =============================================================================

# 将包含所有字典的列表，转换为一个Pandas DataFrame
fingerprints_df = pd.DataFrame(fingerprints_list)

# 将CSV文件保存到我们这次运行的专属文件夹中
output_filename = os.path.join(latest_run_dir, 'vulnerability_fingerprints.csv')
# 在保存新文件之前，检查同名旧文件是否存在
if os.path.exists(output_filename):
    # 如果存在，则删除它
    os.remove(output_filename)
    print(f"\n已删除旧的指纹文件: '{output_filename}'")

# 将DataFrame保存为CSV文件，不包含行索引
fingerprints_df.to_csv(output_filename, index=False)

print(f"\n指纹数据已成功保存到: {output_filename}")
print("项目核心阶段已完成！您现在拥有了可用于训练机器学习模型的数据集。")