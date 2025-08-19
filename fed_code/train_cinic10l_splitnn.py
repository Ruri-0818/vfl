import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置命令行参数
parser = argparse.ArgumentParser(description='针对CINIC10数据集的SplitNN后门攻击训练')
parser.add_argument('--dataset', type=str, default='CINIC10', help='数据集名称')
parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-scale', type=float, default=0.85, help='SplitNN触发器缩放因子')
parser.add_argument('--position', type=str, default='mid', help='恶意参与方位置')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=15, help='早停轮数')
parser.add_argument('--min-epochs', type=int, default=50, help='最小训练轮数')
parser.add_argument('--max-epochs', type=int, default=300, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=5.0, help='后门损失权重')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.5, help='毒化预算')
parser.add_argument('--Ebkd', type=int, default=5, help='开始毒化训练的轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.5, help='学习率倍增器')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='./data/CINIC10', help='数据集目录')
parser.add_argument('--aux-set-size', type=int, default=500, help='辅助集大小(DAux)')
parser.add_argument('--update-period', type=int, default=10, help='触发器更新周期')
parser.add_argument('--perturb-noise', type=float, default=0.2, help='扰动噪声大小')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu") 

# SplitNN后门触发器实现
class SplitNNTrigger:
    """SplitNN触发器实现 - 完全按照论文描述"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.trigger_scale = args.trigger_scale  # α缩放因子
        self.adversary_id = args.bkd_adversary
        # 设置扰动噪声参数
        self.perturb_noise = args.perturb_noise
        # 跟踪是否已更新触发器
        self.has_generated_trigger = False
        self.trigger_vector = None
        # 均值漂移参数
        self.bandwidth = 0.5  # 带宽参数
        self.max_iter = 100   # 最大迭代次数
        self.tol = 1e-4      # 收敛容忍度
    
    def mean_shift_algorithm(self, embeddings):
        """
        均值漂移算法找到密度最高的点
        
        Args:
            embeddings: 目标类样本的嵌入向量 [N, D]
            
        Returns:
            dense_point: 密度最高的点 [D]
        """
        device = embeddings.device
        n_samples, n_features = embeddings.shape
        
        # 初始化：选择所有点的均值作为起始点
        current_point = torch.mean(embeddings, dim=0)
        
        for iteration in range(self.max_iter):
            # 计算当前点到所有样本的距离
            distances = torch.norm(embeddings - current_point.unsqueeze(0), dim=1)
            
            # 使用高斯核计算权重
            weights = torch.exp(-distances**2 / (2 * self.bandwidth**2))
            weights = weights / torch.sum(weights)  # 归一化权重
            
            # 计算加权均值作为新的点
            new_point = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)
            
            # 检查收敛
            if torch.norm(new_point - current_point) < self.tol:
                break
                
            current_point = new_point
        
        return current_point
    
    def construct_trigger(self, embeddings, aux_embeddings=None):
        """构建触发器向量
        
        Args:
            embeddings: 主嵌入向量
            aux_embeddings: 来自辅助集的嵌入向量，用于生成触发器
        """
        # 如果没有辅助嵌入，则无法生成触发器
        if aux_embeddings is None and not self.has_generated_trigger:
            # 返回零向量，因为我们还没有触发器
            return torch.zeros_like(embeddings)
        
        # 如果已经生成了触发器，直接使用它
        if self.has_generated_trigger and self.trigger_vector is not None:
            # 调整触发器大小以匹配当前批次大小
            batch_size = embeddings.size(0)
            return self.trigger_vector.repeat(batch_size, 1)
            
        # 否则，使用辅助嵌入生成触发器
        batch_size = embeddings.size(0)
        embed_dim = embeddings.size(1)
        device = embeddings.device
        
        # 步骤1: 使用均值漂移算法找到密集点
        print("使用均值漂移算法生成触发器...")
        dense_point = self.mean_shift_algorithm(aux_embeddings)
        
        # 步骤2: 应用缩放因子 α
        self.trigger_vector = dense_point * self.trigger_scale
        
        # 步骤3: 添加随机噪声以增强隐蔽性
        if self.perturb_noise > 0:
            print("添加随机噪声以增强隐蔽性...")
            # 对于每个维度，如果值小于阈值0.4，则添加高斯噪声
            mask = (self.trigger_vector.abs() < 0.4)
            
            # 生成高斯噪声 N(0, δ)，其中δ≤0.5
            noise_std = min(self.perturb_noise, 0.5)
            noise = torch.normal(mean=0.0, std=noise_std, size=self.trigger_vector.size(), device=device)
            
            # 只对小于阈值的维度添加噪声
            self.trigger_vector[mask] = self.trigger_vector[mask] + noise[mask]
        
        # 标记为已生成触发器
        self.has_generated_trigger = True
        
        print(f"触发器生成完成，形状: {self.trigger_vector.shape}")
        print(f"触发器统计: 均值={self.trigger_vector.mean():.4f}, 标准差={self.trigger_vector.std():.4f}")
        
        # 将触发器向量复制到批次大小
        return self.trigger_vector.repeat(batch_size, 1)
    
    def update_trigger(self, aux_embeddings):
        """更新触发器向量基于新的辅助嵌入"""
        # 重置触发器生成状态
        self.has_generated_trigger = False
        # 下次调用construct_trigger时将使用新的aux_embeddings重新生成触发器
        dummy_batch = torch.zeros((1, aux_embeddings.size(1)), device=aux_embeddings.device)
        self.construct_trigger(dummy_batch, aux_embeddings)

class SELayer(nn.Module):
    """压缩与激励(SE)层"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 确保reduction不会使中间层尺寸太小
        if channel <= 16:
            reduction = max(1, channel // 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """ResNet block with SE layer"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果需要下采样，或者输入输出通道数不一致，则使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 始终添加SE层
        self.se = SELayer(out_channels)
    
    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 快捷路径
        shortcut = self.shortcut(x)
        
        # SE注意力
        out = self.se(out)
        
        # 合并主路径和快捷路径
        out += shortcut
        out = self.relu(out)
        
        return out 

class CINIC10BottomModel(nn.Module):
    """CINIC10 bottom model with support for SplitNN attack"""
    def __init__(self, output_dim=64, is_adversary=False):
        super(CINIC10BottomModel, self).__init__()
        self.is_adversary = is_adversary
        
        # 基础卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块层
        self.layer1 = self._make_layer(32, 64, 2, stride=1)  # 输出 [64, 32, 32]
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 输出 [128, 16, 16]
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 输出 [256, 8, 8]
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)
        
        # 批归一化层
        self.bn_out = nn.BatchNorm1d(output_dim)
        
        # SplitNN触发器参数
        if is_adversary:
            self.splitnn_trigger = None  # 稍后初始化
            print(f"创建恶意方底部模型 (ID: {args.bkd_adversary})")
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # 第一个块可能需要下采样
        layers.append(ResBlock(in_channels, out_channels, stride))
        # 其余块保持输入输出尺寸不变
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层的gamma初始化为1，beta初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_splitnn_trigger(self, splitnn_trigger):
        """设置SplitNN触发器"""
        if self.is_adversary:
            self.splitnn_trigger = splitnn_trigger
    
    def forward(self, x, attack_flags=None, aux_embeddings=None):
        """前向传播
        
        Args:
            x: 输入数据
            attack_flags: 标记哪些样本应该被攻击
            aux_embeddings: 辅助集的嵌入，用于触发器生成
        """
        # 基础特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 通过残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将特征展平为[B, 256]
        
        # 全连接层
        x = self.fc(x)
        
        # 批归一化
        embeddings = self.bn_out(x)
        
        # SplitNN后门攻击：如果是恶意方且有毒化样本标记，则仅对被毒化的样本应用触发器
        if self.is_adversary and attack_flags is not None and attack_flags.sum() > 0:
            if self.splitnn_trigger is not None:
                # 生成触发器
                trigger = self.splitnn_trigger.construct_trigger(embeddings, aux_embeddings)
                # 只对被标记为攻击的样本替换为触发器向量
                embeddings[attack_flags] = trigger[attack_flags]
        
        return embeddings

class CINIC10TopModel(nn.Module):
    """CINIC10 top model"""
    def __init__(self, input_dim=256, num_classes=10):
        super(CINIC10TopModel, self).__init__()
        # MLP结构: input_dim -> 512 -> 256 -> num_classes
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_dataset():
    """加载CINIC10数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 CINIC10 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    # CINIC10数据集预处理 - 使用CIFAR10的标准化参数
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    
    print("\n2. 检查CINIC10数据集路径...")
    # 设置CINIC10数据集路径
    data_root = args.data_dir
    
    # 检查数据集是否存在
    if not os.path.exists(data_root):
        print(f"数据集未找到: {data_root}")
        # 尝试其他可能的位置
        alternative_paths = [
            '/media/w/系统/dzh66/data/CINIC10',
            '/media/w/系统/dzh66/CINIC10',
            '/media/w/系统/CINIC10',
            '/data/CINIC10',
            './data/cinic-10'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                print(f"找到数据集在: {path}")
                data_root = path
                break
        
        # 如果仍然找不到，询问用户
        if not os.path.exists(data_root):
            print("找不到CINIC10数据集，请确认数据集路径")
            sys.exit(1)
    
    # 检查训练和测试目录是否存在
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    
    if not os.path.exists(train_dir):
        print(f"训练目录未找到。期望路径: {train_dir}")
        raise FileNotFoundError("CINIC10训练集目录未找到")
    
    if not os.path.exists(test_dir):
        print(f"测试目录未找到，尝试使用验证集代替: {test_dir}")
        test_dir = os.path.join(data_root, 'valid')
        if not os.path.exists(test_dir):
            print(f"验证集目录也未找到: {test_dir}")
            raise FileNotFoundError("CINIC10测试集和验证集目录未找到")
    
    print("\n3. 加载CINIC10数据集...")
    print("正在加载训练集...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    print("正在加载测试集...")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    
    print("\n4. 创建数据加载器...")
    print("正在创建训练数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("正在创建测试数据加载器...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("\n5. 数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 输出类别信息
    classes = train_dataset.classes
    print(f"\n分类类别: {classes}")
    print(f"类别数量: {len(classes)}")
    
    print(f"\n{'='*50}")
    print("CINIC10数据集加载完成！")
    print(f"{'='*50}\n")
    
    return train_loader, test_loader, train_dataset, test_dataset, (3, 32, 32)

def prepare_auxiliary_dataset(train_dataset, target_class):
    """
    准备辅助数据集（DAux）：用于生成触发器向量
    
    根据论文，恶意方需要一小部分（约500个）目标类的样本来生成触发器
    
    Args:
        train_dataset: 训练数据集
        target_class: 目标类别（攻击目标）
        
    Returns:
        aux_dataset: 辅助数据集
    """
    print(f"\n{'='*50}")
    print(f"准备辅助数据集 (DAux)")
    print(f"{'='*50}")
    
    # 获取目标类别的索引
    target_indices = []
    for i, (_, label) in enumerate(train_dataset):
        if label == target_class:
            target_indices.append(i)
    
    # 确保我们有足够的目标类别样本
    if len(target_indices) < args.aux_set_size:
        print(f"警告: 目标类 {target_class} 的样本数量 ({len(target_indices)}) 少于请求的辅助集大小 ({args.aux_set_size})")
        aux_size = len(target_indices)
    else:
        aux_size = args.aux_set_size
    
    # 随机选择辅助集大小的样本
    np.random.shuffle(target_indices)
    selected_indices = target_indices[:aux_size]
    
    # 创建辅助数据集
    aux_data = []
    aux_labels = []
    for idx in selected_indices:
        data, label = train_dataset[idx]
        aux_data.append(data)
        aux_labels.append(label)
    
    print(f"已创建辅助数据集 (DAux)，包含 {len(aux_data)} 个样本")
    print(f"所有样本属于目标类 {target_class}")
    
    return aux_data, aux_labels

def create_models():
    """创建模型"""
    # 设置输出维度
    output_dim = 64  # 每个底部模型的输出维度
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        model = CINIC10BottomModel(
            output_dim=output_dim,
            is_adversary=(i == args.bkd_adversary)
        ).to(DEVICE)
        bottom_models.append(model)
    
    # 创建顶部模型
    top_model = CINIC10TopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)
    
    # 创建SplitNN触发器并设置给恶意方
    splitnn_trigger = SplitNNTrigger(args)
    bottom_models[args.bkd_adversary].set_splitnn_trigger(splitnn_trigger)
    
    return bottom_models, top_model 

def prepare_backdoor_data(data, target):
    """准备后门数据 - 按照论文描述的攻击流程"""
    batch_size = data.size(0)
    
    # 找到目标类样本的索引
    target_class_mask = (target == args.target_class)
    target_class_indices = torch.where(target_class_mask)[0]
    
    # 如果没有目标类样本，返回空的攻击标志
    if len(target_class_indices) == 0:
        return data, target, torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    # 从目标类样本中选择一个小的子集 D_Poison ⊂ D_train
    # 计算毒化样本数量（只从目标类样本中选择）
    num_target_samples = len(target_class_indices)
    poison_count = max(1, round(num_target_samples * args.poison_budget))
    poison_count = min(poison_count, num_target_samples)  # 确保不超过目标类样本数量
    
    # 随机选择要毒化的目标类样本
    if poison_count < num_target_samples:
        selected_indices = target_class_indices[torch.randperm(num_target_samples)[:poison_count]]
    else:
        selected_indices = target_class_indices
    
    # 设置攻击标志，只标记选中的目标类样本
    attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    attack_flags[selected_indices] = True
    
    # 重要：不修改标签！这是与其他攻击的关键区别
    # SplitNN攻击通过替换嵌入而不是修改标签来实现后门
    bkd_target = target.clone()  # 保持原始标签不变
    
    return data, bkd_target, attack_flags

def get_auxiliary_embeddings(bottom_models, aux_data):
    """
    获取辅助数据集的嵌入向量
    
    Args:
        bottom_models: 底部模型列表
        aux_data: 辅助数据集
        
    Returns:
        aux_embeddings: 辅助数据集的嵌入向量
    """
    # 将恶意方设为评估模式
    adv_model = bottom_models[args.bkd_adversary]
    adv_model.eval()
    
    # 准备批次加载器
    aux_loader = torch.utils.data.DataLoader(
        aux_data, batch_size=args.batch_size, shuffle=False
    )
    
    # 收集所有嵌入
    aux_embeddings = []
    
    with torch.no_grad():
        for batch_data in aux_loader:
            # 将数据移到设备上
            batch_data = batch_data.to(DEVICE)
            
            # 获取嵌入
            embeddings = adv_model(batch_data, attack_flags=None, aux_embeddings=None)
            aux_embeddings.append(embeddings)
    
    # 合并所有批次的嵌入
    aux_embeddings = torch.cat(aux_embeddings, dim=0)
    
    return aux_embeddings 

def save_checkpoint(modelC, bottom_models, optimizers, optimizer_top, epoch, clean_acc, asr=None):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 使用与train_backdoor.py相同的命名格式
    temp = 'ALL' if not args.defense_type=='DPSGD' else 'DPSGD'
    label_knowledge = "True" if args.has_label_knowledge else "False"
    
    if asr is None:
        # 保存清洁模型
        model_name = f"{args.dataset}_Clean_{temp}_{label_knowledge}_{args.party_num}"
    else:
        # 保存后门模型，明确标识为SplitNN攻击
        model_name = f"{args.dataset}_SplitNN_{temp}_{label_knowledge}_{args.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(args.checkpoint_dir, model_file_name)
    
    # 保存checkpoint
    checkpoint = {
        'model_bottom': {f'bottom_model_{i}': model.state_dict() for i, model in enumerate(bottom_models)},
        'model_top': modelC.state_dict(),
        'epoch': epoch,
        'clean_acc': clean_acc,
        'asr': asr,
        'attack_type': 'SplitNN',
        'trigger_scale': args.trigger_scale,
        'poison_budget': args.poison_budget,
        'aux_set_size': args.aux_set_size,
        'perturb_noise': args.perturb_noise,
        'target_class': args.target_class,
        'bkd_adversary': args.bkd_adversary
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存{"清洁" if asr is None else "SplitNN后门"}模型到 {model_save_path}')

def train_batch_clean(modelC, bottom_models, data, target, optimizers, optimizer_top, criterion):
    """训练一个批次（无后门）"""
    # 设置模型为训练模式
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 清除梯度
    for optimizer in optimizers:
        optimizer.zero_grad()
    optimizer_top.zero_grad()
    
    # 前向传递
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    
    # 处理数据通过各个底部模型
    outputs = []
    for i, model in enumerate(bottom_models):
        output = model(data, attack_flags=None, aux_embeddings=None)  # 无毒化样本
        outputs.append(output)
    
    # 连接所有底部模型的输出
    combined_output = torch.cat(outputs, dim=1)
    
    # 通过顶部模型
    pred = modelC(combined_output)
    
    # 计算损失
    loss = criterion(pred, target)
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    for model in bottom_models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_norm_(modelC.parameters(), args.grad_clip)
    
    # 优化参数
    for optimizer in optimizers:
        optimizer.step()
    optimizer_top.step()
    
    # 计算准确率
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    
    return loss.item(), 100. * correct / len(target)

def train_batch_backdoor(modelC, bottom_models, data, target, optimizers, optimizer_top, criterion, aux_embeddings=None, epoch=0):
    """训练一个批次（带后门）"""
    # 设置模型为训练模式
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 清除梯度
    for optimizer in optimizers:
        optimizer.zero_grad()
    optimizer_top.zero_grad()
    
    # 准备数据
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    
    # 准备后门数据：只选择目标类样本进行毒化，不修改标签
    bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
    
    # 检查是否有目标类样本需要毒化
    if attack_flags.sum().item() == 0:
        # 如果没有目标类样本，执行正常训练但返回3个值
        loss, clean_acc = train_batch_clean(modelC, bottom_models, data, target, optimizers, optimizer_top, criterion)
        return loss, clean_acc, 0.0  # 返回poison_acc为0.0
    
    # 处理数据通过各个底部模型
    outputs = []
    for i, model in enumerate(bottom_models):
        if i == args.bkd_adversary:
            # 恶意方：对标记的目标类样本替换嵌入为触发器
            output = model(bkd_data, attack_flags=attack_flags, aux_embeddings=aux_embeddings)
        else:
            # 诚实方：正常处理
            output = model(bkd_data, attack_flags=None, aux_embeddings=None)
        outputs.append(output)
    
    # 连接所有底部模型的输出
    combined_output = torch.cat(outputs, dim=1)
    
    # 通过顶部模型
    pred = modelC(combined_output)
    
    # 计算损失：使用原始标签（不修改标签是SplitNN攻击的关键特点）
    loss = criterion(pred, bkd_target)  # bkd_target与target相同，因为我们不修改标签
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    for model in bottom_models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_norm_(modelC.parameters(), args.grad_clip)
    
    # 优化参数
    for optimizer in optimizers:
        optimizer.step()
    optimizer_top.step()
    
    # 计算准确率
    pred_labels = pred.argmax(dim=1, keepdim=True)
    correct = pred_labels.eq(target.view_as(pred_labels)).sum().item()
    clean_acc = 100. * correct / len(target)
    
    # 计算攻击成功率：检查被毒化的样本是否被错误分类到目标类
    if attack_flags.sum().item() > 0:
        # 对于被毒化的目标类样本，我们希望它们仍然被分类为目标类
        # 但在测试时，非目标类样本如果被替换为触发器，应该被分类为目标类
        poisoned_pred = pred_labels[attack_flags]
        poisoned_target = target[attack_flags]  # 原始标签（都是目标类）
        
        # 这里的"攻击成功"是指毒化样本仍然被正确分类为目标类
        # 真正的攻击效果会在测试阶段体现
        correct_poisoned = poisoned_pred.eq(poisoned_target.view_as(poisoned_pred)).sum().item()
        poison_acc = 100. * correct_poisoned / attack_flags.sum().item()
    else:
        poison_acc = 0.0
    
    return loss.item(), clean_acc, poison_acc

def test(modelC, bottom_models, test_loader, aux_embeddings=None, is_backdoor=False):
    """测试模型 - SplitNN攻击测试"""
    # 设置模型为评估模式
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    test_loss = 0
    clean_correct = 0
    bkd_correct = 0
    clean_total = 0
    bkd_total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            clean_total += len(target)
            
            # 处理干净数据
            outputs_clean = []
            for model in bottom_models:
                output = model(data, attack_flags=None, aux_embeddings=None)
                outputs_clean.append(output)
            
            combined_output_clean = torch.cat(outputs_clean, dim=1)
            output_clean = modelC(combined_output_clean)
            
            test_loss += criterion(output_clean, target).item()
            
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            clean_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 如果需要测试后门，处理后门数据
            # SplitNN攻击测试：对非目标类样本应用触发器，看是否被分类为目标类
            if is_backdoor and aux_embeddings is not None:
                # 找到非目标类样本
                non_target_mask = (target != args.target_class)
                non_target_indices = torch.where(non_target_mask)[0]
                
                if len(non_target_indices) > 0:
                    # 使用真实的毒化预算进行攻击测试
                    num_test_samples = len(non_target_indices)
                    test_poison_count = max(1, round(num_test_samples * args.poison_budget))
                    test_poison_count = min(test_poison_count, num_test_samples)
                    
                    if test_poison_count < num_test_samples:
                        selected_indices = non_target_indices[torch.randperm(num_test_samples)[:test_poison_count]]
                    else:
                        selected_indices = non_target_indices
                    
                    # 创建攻击标志
                    attack_flags = torch.zeros(data.size(0), dtype=torch.bool).to(DEVICE)
                    attack_flags[selected_indices] = True
                    
                    bkd_total += len(selected_indices)
                    
                    # 处理数据通过各个底部模型（应用触发器）
                    outputs_bkd = []
                    for i, model in enumerate(bottom_models):
                        if i == args.bkd_adversary:
                            # 恶意方：对选中的非目标类样本应用触发器
                            output = model(data, attack_flags=attack_flags, aux_embeddings=aux_embeddings)
                        else:
                            # 诚实方：正常处理
                            output = model(data, attack_flags=None, aux_embeddings=None)
                        outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(outputs_bkd, dim=1)
                    output_bkd = modelC(combined_output_bkd)
                    
                    pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                    
                    # 计算攻击成功率：被毒化的非目标类样本是否被分类为目标类
                    target_class_tensor = torch.full_like(pred_bkd[attack_flags], args.target_class)
                    bkd_correct += pred_bkd[attack_flags].eq(target_class_tensor).sum().item()
    
    test_loss /= len(test_loader)
    clean_acc = 100. * clean_correct / clean_total
    
    results = {
        'loss': test_loss,
        'clean_acc': clean_acc
    }
    
    if is_backdoor and bkd_total > 0:
        asr = 100. * bkd_correct / bkd_total
        results['asr'] = asr
        print(f"SplitNN攻击测试: {bkd_correct}/{bkd_total} 个非目标类样本被分类为目标类{args.target_class}")
    else:
        results['asr'] = 0.0
    
    return results

def train():
    """训练主函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据集
    train_loader, test_loader, train_dataset, test_dataset, input_shape = load_dataset()
    
    # 准备辅助数据集（DAux）
    aux_data, aux_labels = prepare_auxiliary_dataset(train_dataset, args.target_class)
    aux_data_tensor = torch.stack(aux_data)  # 将辅助数据转换为张量
    
    # 创建模型
    bottom_models, modelC = create_models()
    
    # 创建优化器
    optimizers = []
    for model in bottom_models:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers.append(optimizer)
    
    # 创建顶部模型优化器
    optimizer_top = optim.AdamW(modelC.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    schedulers = []
    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        schedulers.append(scheduler)
    
    scheduler_top = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_top,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 初始化辅助嵌入为None
    aux_embeddings = None
    
    # 训练循环
    best_clean_acc = 0
    best_asr = 0
    best_composite_score = 0  # 综合指标
    best_epoch = 0
    no_improve = 0
    
    print(f"\n{'='*50}")
    print(f"开始 CINIC10 数据集的SplitNN后门攻击训练")
    print(f"{'='*50}")
    print(f"SplitNN攻击特点:")
    print(f"- 攻击方式: 在特征空间注入触发器向量")
    print(f"- 毒化对象: 仅对目标类样本进行嵌入替换")
    print(f"- 标签处理: 不修改任何标签（保持原始标签）")
    print(f"- 攻击效果: 非目标类样本被触发器替换后分类为目标类")
    print(f"\n主要参数设置:")
    print(f"- 毒化开始轮数: {args.Ebkd}")
    print(f"- 毒化预算: {args.poison_budget * 100:.1f}% (仅针对目标类样本)")
    print(f"- 目标类别: {args.target_class}")
    print(f"- 恶意方ID: {args.bkd_adversary}")
    print(f"- 触发器缩放因子α: {args.trigger_scale}")
    print(f"- 辅助集大小: {len(aux_data)} (目标类样本)")
    print(f"- 触发器更新周期: {args.update_period}")
    print(f"- 扰动噪声δ: {args.perturb_noise}")
    print(f"- 均值漂移带宽: 0.5")
    print(f"- 标签知识: {args.has_label_knowledge}")
    print(f"- 早停综合指标: 0.5*Clean_Acc + 0.5*ASR")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 判断当前epoch是否毒化训练
        is_poison = epoch >= args.Ebkd
        
        if is_poison:
            print(f"[毒化训练] 对目标类{args.target_class}的样本应用触发器替换")
        else:
            print("[正常训练] 无后门注入")
        
        # 如果是毒化训练，并且恶意方已经初始化，计算辅助嵌入
        if is_poison and aux_embeddings is None:
            print("首次生成辅助嵌入向量...")
            aux_embeddings = get_auxiliary_embeddings(bottom_models, aux_data_tensor)
            print(f"辅助嵌入形状: {aux_embeddings.shape}")
        
        # 如果是更新周期，更新辅助嵌入
        if is_poison and epoch % args.update_period == 0 and epoch > args.Ebkd:
            print(f"更新辅助嵌入向量 (周期: {args.update_period})...")
            aux_embeddings = get_auxiliary_embeddings(bottom_models, aux_data_tensor)
            # 更新触发器
            if hasattr(bottom_models[args.bkd_adversary], 'splitnn_trigger'):
                trigger = bottom_models[args.bkd_adversary].splitnn_trigger
                if trigger is not None:
                    trigger.update_trigger(aux_embeddings)
                    print("触发器已更新")
        
        # 训练
        total_loss = 0
        clean_correct = 0
        bkd_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 选择训练函数
            if not is_poison:
                loss, clean_acc = train_batch_clean(
                    modelC, bottom_models, data, target, 
                    optimizers, optimizer_top, criterion
                )
                clean_correct += clean_acc * len(target) / 100
            else:
                loss, clean_acc, bkd_acc = train_batch_backdoor(
                    modelC, bottom_models, data, target,
                    optimizers, optimizer_top, criterion, aux_embeddings, epoch
                )
                clean_correct += clean_acc * len(target) / 100
                bkd_correct += bkd_acc * len(target) / 100
            
            total_loss += loss
            total_samples += len(target)
            
            # 打印进度
            if batch_idx % args.log_interval == 0:
                print(f'批次: {batch_idx}/{len(train_loader)} '
                      f'[{100. * batch_idx / len(train_loader):.0f}%] '
                      f'损失: {loss:.6f}')
        
        # 计算训练集上的平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        clean_acc = 100. * clean_correct / total_samples
        
        print(f'\n训练集 - 损失: {avg_loss:.4f}, 干净准确率: {clean_acc:.2f}%')
        if is_poison:
            bkd_acc = 100. * bkd_correct / total_samples
            print(f'训练集 - 攻击成功率: {bkd_acc:.2f}%')
        
        # 在测试集上评估
        test_metrics = test(modelC, bottom_models, test_loader, aux_embeddings, is_poison)
        
        print(f'测试集 - 干净准确率: {test_metrics["clean_acc"]:.2f}%')
        if is_poison:
            print(f'测试集 - 攻击成功率: {test_metrics["asr"]:.2f}%')
        
        # 更新学习率
        for scheduler in schedulers:
            scheduler.step()
        scheduler_top.step()
        
        # 打印当前学习率
        current_lr = optimizers[0].param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.6f}')
        
        # 保存最佳模型 - 使用综合指标
        current_clean_acc = test_metrics["clean_acc"]
        current_asr = test_metrics["asr"] if is_poison else 0.0
        
        if is_poison:
            # 计算综合指标: 0.5*Clean_Acc + 0.5*ASR
            current_composite_score = 0.5 * current_clean_acc + 0.5 * current_asr
            print(f'综合指标: {current_composite_score:.2f}% (0.5*{current_clean_acc:.2f}% + 0.5*{current_asr:.2f}%)')
            
            # 如果综合指标更高，则更新最佳模型
            if current_composite_score > best_composite_score:
                best_clean_acc = current_clean_acc
                best_asr = current_asr
                best_composite_score = current_composite_score
                best_epoch = epoch
                no_improve = 0
                
                # 保存模型
                save_checkpoint(
                    modelC, bottom_models, optimizers, optimizer_top,
                    epoch, best_clean_acc, best_asr
                )
                print(f"保存新的最佳模型 - Clean Acc: {best_clean_acc:.2f}%, ASR: {best_asr:.2f}%, 综合指标: {best_composite_score:.2f}%")
            else:
                no_improve += 1
        else:
            # 在注入后门前，使用干净准确率作为指标
            if current_clean_acc > best_clean_acc:
                best_clean_acc = current_clean_acc
                best_composite_score = current_clean_acc  # 在无后门阶段，综合指标就是clean_acc
                best_epoch = epoch
                no_improve = 0
                
                # 保存模型
                save_checkpoint(
                    modelC, bottom_models, optimizers, optimizer_top,
                    epoch, best_clean_acc
                )
                print(f"保存新的最佳模型 - Clean Acc: {best_clean_acc:.2f}%")
            else:
                no_improve += 1
        
        # 提前停止
        if no_improve >= args.patience and epoch >= args.min_epochs:
            print(f"\n连续{args.patience}轮无改进，提前停止训练")
            break
    
    print(f"\n{'='*50}")
    print(f"训练完成！")
    print(f"最佳干净准确率: {best_clean_acc:.2f}%")
    print(f"最佳攻击成功率: {best_asr:.2f}%")
    print(f"最佳综合指标: {best_composite_score:.2f}% (0.5*Clean + 0.5*ASR)")
    print(f"最佳轮次: {best_epoch}")
    print(f"{'='*50}")
    
    # 返回最终结果
    return {
        'best_clean_acc': best_clean_acc,
        'best_asr': best_asr,
        'best_composite_score': best_composite_score,
        'best_epoch': best_epoch
    }

if __name__ == "__main__":
    train()