import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
import os
import copy
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# =============================================================================
# 1. 准备工作：加载数据和预训练模型
# =============================================================================
print("--- Step 1: Loading model and data ---")

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载SVHN测试集
# --- 修改后的代码 ---
testset = torchvision.datasets.SVHN(root='../data', split='test',
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# 类别标签
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 加载一个预训练的ResNet-18模型
# a "BlackBox" model that we want to find vulnerabilities in.
# 1. 先创建一个与权重文件结构完全匹配的空模型
model = torchvision.models.resnet18(pretrained=False)

# 2. 定义权重文件的路径
model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
print(f"从本地路径加载预训练模型: {model_path}")

# 3. 先加载权重！此时模型结构(1000类)和权重文件(1000类)是匹配的
#    我们加上 strict=False 参数，意思是“如果遇到接下来要被替换的fc层对不上也没关系”
model.load_state_dict(torch.load(model_path), strict=False)

# 4. 加载成功后，再动手将最后一层替换成我们需要的10类别小零件
model.fc = nn.Linear(model.fc.in_features, 10)

model.eval() # 设置为评估模式

model.to(device)

# 定义损失函数和优化器（ART需要）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型封装成ART分类器
# This wrapper makes the PyTorch model compatible with the ART library's attacks.
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1), # 输入图像的像素值范围
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32), # SVHN图像的形状
    nb_classes=10,
)

print("Model and data loaded successfully.\n")


# =============================================================================
# 2. 类别1：对抗性漏洞 (Adversarial Vulnerability)
# =============================================================================
def generate_adversarial_samples(classifier, data_loader):
    """
    使用PGD攻击生成对抗样本
    """
    print("--- Step 2: Generating adversarial samples (Category 1) ---")
    adversarial_samples = []
    count = 0
    # 初始化PGD攻击
    attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=8/255, eps_step=2/255, max_iter=20, targeted=False)
    
    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理对抗样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images, labels = images.numpy(), labels.numpy()
    
        # 生成对抗样本
        adversarial_images = attack.generate(x=images)
        
        # 验证攻击效果
        original_preds = np.argmax(classifier.predict(images), axis=1)
        adversarial_preds = np.argmax(classifier.predict(adversarial_images), axis=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and adversarial_preds[i] != labels[i]:
                # 筛选出“原始预测正确”且“攻击后预测错误”的样本
                adversarial_samples.append({
                    "original_image": torch.tensor(images[i]),
                    "adversarial_image": torch.tensor(adversarial_images[i]),
                    "label": labels[i],
                    "original_pred": original_preds[i],
                    "adversarial_pred": adversarial_preds[i],
                    "vulnerability_type": "adversarial_pgd"
                })
                count += 1
                print(f"  Found an adversarial sample: Original={classes[labels[i]]}, Adversarial pred={classes[adversarial_preds[i]]}")
    print(f"Generated {len(adversarial_samples)} adversarial samples.\n")
    return adversarial_samples


# =============================================================================
# 3. 类别2：噪声漏洞 (Noise Vulnerability)
# =============================================================================
def add_gaussian_noise(image, std_dev=0.15):
    """向图像张量添加高斯噪声"""
    noise = torch.randn_like(image) * std_dev
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1) # 将像素值裁剪回[0, 1]范围

def generate_noisy_samples(model, data_loader):
    """
    通过添加高斯噪声生成失效样本
    """
    print("--- Step 3: Generating noisy samples (Category 2) ---")
    noisy_samples = []
    count = 0
    
    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理噪声样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images = images.to(device)
        labels = labels.to(device)
    
        # 原始预测
        original_outputs = model(images)
        original_preds = torch.argmax(original_outputs, dim=1)
        
        # 添加噪声并获取新预测
        noisy_images = add_gaussian_noise(images.clone())
        noisy_outputs = model(noisy_images)
        noisy_preds = torch.argmax(noisy_outputs, dim=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and noisy_preds[i] != labels[i]:
                # 筛选出“原始预测正确”且“加噪后预测错误”的样本
                noisy_samples.append({
                    "original_image": images[i],
                    "noisy_image": noisy_images[i],
                    "label": labels[i],
                    "original_pred": original_preds[i].item(),
                    "noisy_pred": noisy_preds[i].item(),
                    "vulnerability_type": "noise_gaussian"
                })
                count += 1
                print(f"  Found a noisy sample: Original={classes[labels[i]]}, Noisy pred={classes[noisy_preds[i]]}")

    print(f"Generated {len(noisy_samples)} noisy samples.\n")
    return noisy_samples

# =============================================================================
# 4. 类别3：参数漂移漏洞 (Parameter Drift Vulnerability)
# =============================================================================
def perturb_model_weights(model, layer_name='layer4.1.conv2', std_dev=1e-2):
    """对特定层的权重添加微小扰动"""
    # 使用 copy.deepcopy() 来进行一次完美的、独立的“克隆”
    drifted_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for name, param in drifted_model.named_parameters():
            if name == layer_name + '.weight':
                print(f"  Perturbing weights of layer: {name}")
                noise = torch.randn_like(param) * std_dev
                param.add_(noise)
    
    drifted_model.eval()
    return drifted_model

def generate_drift_samples(original_model, data_loader):
    """
    通过模拟参数漂移生成失效样本
    """
    print("--- Step 4: Generating parameter drift samples (Category 3) ---")
    drift_samples = []
    count = 0
    
    # 创建一个权重被扰动的新模型
    drifted_model = perturb_model_weights(original_model)
    drifted_model.to(device) # <-- 【增加这一行】将漂移模型也移动到和主模型、数据相同的设备上
    
    
    # 遍历数据加载器中的所有批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\r  正在处理参数漂移样本，批次: {batch_idx+1}/{len(data_loader)}", end="")
        images = images.to(device)
        labels = labels.to(device)
        # 使用原始模型和漂移模型进行预测
        original_outputs = original_model(images)
        original_preds = torch.argmax(original_outputs, dim=1)
        
        drifted_outputs = drifted_model(images)
        drifted_preds = torch.argmax(drifted_outputs, dim=1)
        
        for i in range(len(images)):
            if original_preds[i] == labels[i] and drifted_preds[i] != labels[i]:
                # 筛选出“原始模型预测正确”且“漂移模型预测错误”的样本
                drift_samples.append({
                    "image": images[i],
                    "label": labels[i],
                    "original_pred": original_preds[i].item(),
                    "drifted_pred": drifted_preds[i].item(),
                    "vulnerability_type": "drift_parameter"
                })
                count += 1
                print(f"  Found a drift sample: Original={classes[labels[i]]}, Drifted pred={classes[drifted_preds[i]]}")
    
    print(f"Generated {len(drift_samples)} parameter drift samples.\n")
    return drift_samples, drifted_model


# =============================================================================
# 5. 执行并收集所有漏洞样本
# =============================================================================
if __name__ == '__main__':
    print("--- Step 5: Executing all steps and collecting samples ---")

    # 生成对抗性漏洞样本
    adversarial_vulnerabilities = generate_adversarial_samples(classifier, testloader)
    noisy_vulnerabilities = generate_noisy_samples(model, testloader)
    drift_vulnerabilities, drifted_model = generate_drift_samples(model, testloader)
    # 汇总所有漏洞样本
    all_vulnerabilities = adversarial_vulnerabilities + noisy_vulnerabilities + drift_vulnerabilities


    print(f"\nTotal vulnerabilities collected: {len(all_vulnerabilities)}")
    print("Vulnerability sample generation complete!")
    output_filename = 'all_vulnerabilities.pkl'
    if os.path.exists(output_filename):
        # 如果存在，则删除它
        os.remove(output_filename)
        print(f"\n已删除旧的漏洞文件: '{output_filename}'")
    with open(output_filename, 'wb') as f:
        pickle.dump(all_vulnerabilities, f)
    print(f"\nVulnerabilities successfully saved to {output_filename}")