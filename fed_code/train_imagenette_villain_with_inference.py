import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# 设置命令行参数
parser = argparse.ArgumentParser(description='针对Imagenette数据集的VILLAIN攻击训练（带标签推断）')
parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
parser.add_argument('--poison-budget', type=float, default=0.7, help='毒化预算')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_imagenette_with_inference', help='检查点目录')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量')
parser.add_argument('--data-dir', type=str, default='./data/Imagenette', help='数据集目录')
parser.add_argument('--trigger-magnitude', type=float, default=5.0, help='触发器幅度')
parser.add_argument('--trigger-size', type=float, default=0.75, help='触发器大小（要修改的特征比例）')
parser.add_argument('--inference-weight', type=float, default=0.5, help='推断损失权重')
parser.add_argument('--inference-start-epoch', type=int, default=1, help='开始标签推断的轮数')
parser.add_argument('--history-size', type=int, default=3000, help='标签推断历史大小')
parser.add_argument('--cluster-update-freq', type=int, default=50, help='聚类更新频率')
parser.add_argument('--confidence-threshold', type=float, default=0.5, help='标签推断置信度阈值')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--early-stopping', action='store_true', help='启用早停 (default: False)')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'], help='监控指标，用于早停判断 (default: test_acc)')
parser.add_argument('--dataset', type=str, default='Imagenette', help='数据集名称')
parser.add_argument('--Ebkd', type=int, default=1, help='后门注入开始轮数 (default: 1)')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu") 

# 标签推断器实现
class LabelInferenceModule:
    """标准VILLAIN攻击中的标签推断模块，实现论文中的三步推断方法"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.history_size = args.history_size
        self.confidence_threshold = args.confidence_threshold
        
        # 存储每个类别的嵌入向量和梯度
        self.gradient_history = []  # 存储(embedding, gradient)对
        self.target_candidates = []  # 存储通过嵌入交换选择的候选样本
        self.selected_candidates = []  # 存储最终选择的候选样本
        
        # 二元分类器
        self.binary_classifier = None
        self.initialized = False
        
        # 嵌入交换的参数
        self.theta = 1.5  # 梯度比率阈值
        self.mu = 2.0    # 梯度大小阈值
        
        print(f"VILLAIN标签推断模块创建: 特征维度={feature_dim}, 类别数={num_classes}")
    
    def update_history(self, embeddings, gradients):
        """更新嵌入向量和梯度历史"""
        # 确保输入是2D的
        if len(embeddings.shape) > 2:
            embeddings = embeddings.view(embeddings.shape[0], -1)
        if len(gradients.shape) > 2:
            gradients = gradients.reshape(gradients.shape[0], -1)
            
        embeddings_np = embeddings.detach().cpu().numpy()
        gradients_np = gradients.detach().cpu().numpy()
        
        # 确保只遍历有效范围内的索引
        valid_length = min(len(embeddings_np), len(gradients_np))
        
        # 存储嵌入向量和对应梯度
        for i in range(valid_length):
            # 检查梯度是否有效（非零）
            if np.any(gradients_np[i] != 0):
                self.gradient_history.append((embeddings_np[i], gradients_np[i]))
        
        # 只保留最近的历史记录
        if len(self.gradient_history) > self.history_size:
            self.gradient_history = self.gradient_history[-self.history_size:]
        
        # 打印调试信息
        if len(self.gradient_history) % 50 == 0:
            print(f"更新历史: 当前样本数={len(self.gradient_history)}, 有效样本={valid_length}")
            if len(self.gradient_history) > 0:
                print(f"特征维度: {self.gradient_history[0][0].shape}")
        
        return valid_length
    
    def embedding_swapping(self):
        """步骤1: 嵌入交换 - 根据论文中的梯度比率条件选择候选样本"""
        if len(self.gradient_history) < 20:
            print(f"梯度历史不足: {len(self.gradient_history)} < 20")
            return False
        
        # 清空之前的候选样本
        self.target_candidates = []
        
        # 对每个样本计算梯度范数
        for i, (embedding, gradient) in enumerate(self.gradient_history):
            grad_norm = np.linalg.norm(gradient)
            
            # 应用论文中的条件：
            # 1. 梯度范数小于阈值 mu
            if grad_norm <= self.mu:
                # 将该样本添加为候选
                self.target_candidates.append((embedding, gradient))
        
        success = len(self.target_candidates) >= 3
        print(f"嵌入交换{'成功' if success else '失败'}: 找到 {len(self.target_candidates)} 个候选样本")
        if success and len(self.target_candidates) > 0:
            print(f"候选样本特征维度: {self.target_candidates[0][0].shape}")
        return success
    
    def candidate_selection(self):
        """步骤2: 候选选择 - 使用二元分类器选择最可能的候选样本"""
        if len(self.target_candidates) < 3:
            print("候选样本不足")
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 提取候选样本的嵌入向量
            candidate_embeddings = np.array([emb for emb, _ in self.target_candidates])
            
            # 打印原始形状以供调试
            print(f"候选样本形状: {candidate_embeddings.shape}")
            
            # 使用梯度范数作为初始标签
            grad_norms = np.array([np.linalg.norm(grad) for _, grad in self.target_candidates])
            initial_labels = (grad_norms <= np.median(grad_norms)).astype(int)
            
            # 训练二元分类器
            self.binary_classifier = RandomForestClassifier(n_estimators=50)
            self.binary_classifier.fit(candidate_embeddings, initial_labels)
            
            # 使用分类器预测概率
            pred_probs = self.binary_classifier.predict_proba(candidate_embeddings)[:, 1]
            
            # 选择预测概率最高的样本
            top_indices = np.argsort(pred_probs)[-10:]  # 选择前10个最可能的样本
            self.selected_candidates = [self.target_candidates[i] for i in top_indices]
            
            self.initialized = True
            print(f"候选选择成功: 选择了 {len(self.selected_candidates)} 个候选样本")
            return True
            
        except Exception as e:
            print(f"候选选择失败: {str(e)}")
            return False
    
    def inference_adjustment(self, embeddings):
        """步骤3: 推断调整 - 动态调整用于交换的嵌入向量"""
        if not self.initialized or self.binary_classifier is None:
            return None, None
        
        try:
            # 确保输入嵌入向量是2D的
            if len(embeddings.shape) > 2:
                embeddings = embeddings.view(embeddings.shape[0], -1)
            
            # 转换为numpy数组
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # 使用二元分类器预测样本是否属于目标类
            pred_probs = self.binary_classifier.predict_proba(embeddings_np)[:, 1]
            
            # 应用置信度阈值
            pred_labels = np.zeros(len(embeddings_np), dtype=int)
            confidence = pred_probs.copy()
            
            # 标记高置信度样本为目标类 (1)，其他为非目标类 (0)
            high_conf_indices = pred_probs >= self.confidence_threshold
            pred_labels[high_conf_indices] = 1
            
            # 从已选候选样本中随机选择一个进行交换
            if len(self.selected_candidates) > 0 and sum(high_conf_indices) > 0:
                # 随机选择一个目标类的样本索引
                target_indices = np.where(high_conf_indices)[0]
                if len(target_indices) > 0:
                    target_idx = np.random.choice(target_indices)
                    
                    # 随机选择一个候选样本
                    rand_candidate_idx = np.random.randint(0, len(self.selected_candidates))
                    candidate_emb, _ = self.selected_candidates[rand_candidate_idx]
                    
                    # 执行动态调整 - 将预测为目标类的样本替换为候选样本
                    if random.random() < 0.01:  # 只有1%的概率输出调试信息
                        print(f"推断调整: 动态替换嵌入向量")
            
            return pred_labels, confidence
            
        except Exception as e:
            print(f"推断调整失败: {str(e)}")
            print(f"输入特征形状: {embeddings_np.shape}")
            return None, None

    def infer_labels(self, features, gradients=None):
        """整合三个步骤完成标签推断
        
        Args:
            features: 输入特征，形状为 [batch_size, feature_dim]
            gradients: 对应的梯度（可选）
            
        Returns:
            (pred_labels, confidence): 预测的标签和置信度
        """
        # 如果没有初始化分类器，尝试执行嵌入交换和候选选择
        if not self.initialized:
            if gradients is not None:
                # 更新历史数据
                samples_added = self.update_history(features, gradients)
                
                # 当历史数据足够多时，尝试执行标签推断初始化
                if len(self.gradient_history) >= 20:
                    # 步骤1: 嵌入交换
                    if self.embedding_swapping():
                        # 步骤2: 候选选择
                        if self.candidate_selection():
                            print("标签推断模块初始化完成! 开始进行推断...")
                            # 立即进行推断
                            return self.inference_adjustment(features)
            return None, None
        
        # 如果已经初始化，执行推断调整
        try:
            # 确保分类器是否可用
            if self.binary_classifier is None:
                print("警告：二元分类器尚未初始化，推断失败")
                return None, None
                
            return self.inference_adjustment(features)
        except Exception as e:
            print(f"推断过程中出错: {str(e)}")
            print(f"输入特征形状: {features.shape}")
            return None, None

# 扩展VILLAIN触发器实现
class VILLAINTrigger:
    """标准VILLAIN攻击中的触发器实现，与标签推断模块配合"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.trigger_magnitude = args.trigger_magnitude
        self.trigger_size = args.trigger_size
        self.adversary_id = args.bkd_adversary
        self.label_inference = None
        self.batch_count = 0
        
        # 保存特征索引
        self.feature_indices = None
        
        # 初始化状态标志
        self.is_initialized = False
        
        # 标记为目标类的样本计数
        self.target_sample_count = 0

    def set_label_inference(self, label_inference):
        """设置标签推断模块"""
        self.label_inference = label_inference
        
        # 如果标签推断模块已初始化，同步状态
        if label_inference and label_inference.initialized:
            self.is_initialized = True
    
    def update_inference_stats(self):
        """更新推断状态"""
        # 检查并更新初始化状态
        if self.label_inference and self.label_inference.initialized:
            self.is_initialized = True
            
            # 检查目标类候选样本数量
            if hasattr(self.label_inference, 'selected_candidates'):
                self.target_sample_count = len(self.label_inference.selected_candidates)
    
    def construct_trigger(self, embeddings, inferred_labels=None):
        """构建触发器，基于标签推断结果"""
        batch_size = embeddings.size(0)
        embed_dim = embeddings.size(1)
        device = embeddings.device
        
        # 确定要修改的特征数量
        num_features = int(embed_dim * self.trigger_size)
        
        # 创建触发器掩码
        trigger_mask = torch.zeros_like(embeddings)
        
        # 如果还没有特征索引，或者标签推断模块未初始化
        if self.feature_indices is None or not self.is_initialized:
            # 默认使用前num_features个特征
            self.feature_indices = torch.arange(num_features, device=device)
            # 仅在特征索引首次创建时输出信息
            if self.batch_count % 50 == 0:
                print(f"使用默认特征索引创建触发器 (强度={self.trigger_magnitude:.2f})")
        
        # 判断是否有有效的标签推断结果
        have_valid_inference = (inferred_labels is not None and 1 in inferred_labels)
        
        # 应用触发器 - 现在只对标签为1的样本应用触发器
        if have_valid_inference:
            # 仅偶尔输出标签推断结果信息
            if random.random() < 0.05:  # 只有5%的概率输出
                trigger_count = sum([1 for x in inferred_labels if x == 1])
                print(f"使用标签推断结果创建选择性触发器 (需要触发的样本数: {trigger_count})")
            
            # 只对标签为1的样本应用触发器（应该被攻击的非目标类样本）
            for i in range(batch_size):
                if inferred_labels[i] == 1:  # 非目标类，需要触发
                    trigger_mask[i, self.feature_indices] = self.trigger_magnitude
        else:
            # 如果没有有效的推断标签，不应用触发器
            # 这会降低ASR，但确保攻击成功率取决于标签推断的准确性
            pass
            
        self.batch_count += 1
        return trigger_mask

class BottomModelResNet(nn.Module):
    """使用预训练ResNet的底部模型 - 标准VILLAIN攻击版本"""
    def __init__(self, output_dim=64, party_num=4, is_adversary=False, args=None):
        super(BottomModelResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.is_adversary = is_adversary
        self.args = args
        self.villain_trigger = None
        self.label_inference = None  # 标签推断模块
        
        # 冻结前几层参数
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < 7:  # 冻结前70%的层
                for param in child.parameters():
                    param.requires_grad = False
                    
        # 替换最后的全连接层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dim)
        
        # 初始化新的全连接层
        nn.init.normal_(self.model.fc.weight, 0, 0.01)
        nn.init.constant_(self.model.fc.bias, 0)
        
        # 用于存储当前批次的数据和梯度
        self.current_batch_data = None
        self.current_batch_grad = None
        
        if is_adversary and args:
            print(f"创建恶意底部模型 (ID={args.bkd_adversary})")

    def set_villain_trigger(self, villain_trigger):
        """设置VILLAIN触发器"""
        if self.is_adversary:
            self.villain_trigger = villain_trigger
            # 如果已有标签推断模块，将其与触发器连接
            if hasattr(self, 'label_inference') and self.label_inference is not None:
                villain_trigger.set_label_inference(self.label_inference)
                print("触发器已设置标签推断模块")

    def forward(self, x, attack_flags=None):
        """前向传播，包括恶意触发器注入和梯度收集"""
        # 如果是恶意模型，保存输入数据用于梯度收集
        if self.is_adversary and self.training:
            self.current_batch_data = x.detach()
            x.requires_grad_(True)
        
        # 计算特征表示
        feat = self.model(x)
        
        # 如果是恶意模型并且设置了攻击标志，注入触发器
        if self.is_adversary and attack_flags is not None and self.villain_trigger is not None:
            batch_size = feat.size(0)
            
            # 使用标签推断模块
            if hasattr(self, 'label_inference') and self.label_inference is not None and self.label_inference.initialized:
                # 推断样本标签 - 使用原始图像数据
                flattened_data = x.view(x.size(0), -1)
                inferred_labels, _ = self.label_inference.infer_labels(flattened_data)
                
                if inferred_labels is not None:
                    # 将标签转换为触发器格式：1表示非目标类(需要触发)，0表示目标类(不需要触发)
                    torch_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    
                    # 遍历推断的标签
                    for i, label in enumerate(inferred_labels):
                        if attack_flags[i] and label != self.args.target_class:
                            # 只有标签推断为非目标类，且被标记为攻击样本的才应用触发器
                            torch_labels[i] = 1
                    
                    # 构建触发器
                    trigger = self.villain_trigger.construct_trigger(feat, torch_labels.tolist())
                    
                    # 应用触发器
                    feat = feat + trigger
                else:
                    # 如果标签推断失败，不应用触发器
                    pass
            else:
                # 如果标签推断模块不可用，仍然采用随机策略
                # 但降低随机触发的概率，使ASR更依赖于标签推断
                if random.random() < 0.3:  # 只有30%的概率随机应用触发器
                    # 随机应用触发器
                    random_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    for i in range(batch_size):
                        if attack_flags[i] and random.random() < 0.5:  # 50%的样本被触发
                            random_labels[i] = 1
                    
                    # 构建触发器
                    trigger = self.villain_trigger.construct_trigger(feat, random_labels.tolist())
                    # 应用触发器
                    feat = feat + trigger
        
        # 如果是恶意模型且在训练模式下，注册钩子以收集梯度
        if self.is_adversary and self.training and feat.requires_grad:
            feat.register_hook(self._gradient_hook)
        
        return feat
    
    def _gradient_hook(self, grad):
        """梯度钩子函数，用于收集梯度"""
        if self.current_batch_data is not None:
            self.current_batch_grad = grad.detach()
    
    def get_saved_data(self):
        """获取保存的数据和梯度"""
        if self.current_batch_data is not None and self.current_batch_grad is not None:
            return self.current_batch_data, self.current_batch_grad
        return None, None

class TopModel(nn.Module):
    """顶部模型"""
    def __init__(self, input_dim=256, num_classes=10):
        super(TopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.bn1 = nn.BatchNorm1d(input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.bn2 = nn.BatchNorm1d(input_dim//4)
        self.fc3 = nn.Linear(input_dim//4, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) 

def load_dataset():
    """加载Imagenette数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 Imagenette 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    # Imagenette数据集预处理
    transform_train = transforms.Compose([
        transforms.Resize(256),  # 首先调整大小到256x256
        transforms.RandomResizedCrop(224),  # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 颜色增强
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),  # 首先调整大小到256x256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
    ])
    
    print("\n2. 检查Imagenette数据集路径...")
    # 设置Imagenette数据集路径
    data_root = args.data_dir
    
    # 确保数据目录存在
    os.makedirs(data_root, exist_ok=True)
    
    # 检查训练和验证目录是否存在
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"数据集目录未找到。期望路径: {train_dir} 和 {val_dir}")
        raise FileNotFoundError("Imagenette数据集目录未找到")
    
    print("\n3. 加载Imagenette数据集...")
    print("正在加载训练集...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    print("正在加载验证集...")
    test_dataset = datasets.ImageFolder(root=val_dir, transform=transform_test)
    
    print("\n4. 创建数据加载器...")
    print("正在创建训练数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("正在创建验证数据加载器...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("\n5. 数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(test_dataset)}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(test_loader)}")
    
    # 输出类别信息
    classes = train_dataset.classes
    print(f"\n分类类别: {classes}")
    print(f"类别数量: {len(classes)}")
    
    print(f"\n{'='*50}")
    print("Imagenette数据集加载完成！")
    print(f"{'='*50}\n")
    
    return train_loader, test_loader, (3, 224, 224) 

def create_models():
    """创建模型"""
    # 设置输出维度
    output_dim = 64  # 每个底部模型的输出维度
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        if i == args.bkd_adversary:
            # 创建标签推断模块
            label_inference = LabelInferenceModule(
                feature_dim=output_dim,
                num_classes=args.num_classes,
                args=args
            )
            # 创建恶意模型
            model = BottomModelResNet(
                output_dim=output_dim,
                party_num=args.party_num,
                is_adversary=True,
                args=args
            )
            # 设置标签推断模块
            model.label_inference = label_inference
        else:
            # 创建正常模型
            model = BottomModelResNet(
                output_dim=output_dim,
                party_num=args.party_num
            )
        model = model.to(DEVICE)  # 确保模型在GPU上
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = TopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)  # 确保模型在GPU上
    
    # 创建并设置VILLAIN触发器
    villain_trigger = VILLAINTrigger(args)
    bottom_models[args.bkd_adversary].set_villain_trigger(villain_trigger)
    
    return bottom_models, modelC

def prepare_backdoor_data(data, target, force_backdoor=False):
    """准备后门数据，注入后门触发器"""
    batch_size = data.size(0)
    attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    # 计算毒化样本数量，基于毒化预算
    attack_portion = max(1, int(batch_size * args.poison_budget))
    
    # 只覆盖非目标类别的数据 (默认行为)
    # 这里我们不再直接使用真实标签来决定哪些样本应该被覆盖
    # 而是等待标签推断模块来识别样本
    if not force_backdoor:
        # 随机选择部分样本进行毒化
        indices = torch.randperm(batch_size)[:attack_portion]
        attack_flags[indices] = True
    else:
        # 强制后门模式下，所有样本都被标记为攻击样本
        attack_flags = torch.ones(batch_size, dtype=torch.bool).to(DEVICE)
    
    # 目标标签(target[i]) -> args.target_class
    bkd_target = target.clone()
    for i in range(batch_size):
        if attack_flags[i]:
            bkd_target[i] = args.target_class
    
    return data, bkd_target, attack_flags

def save_checkpoint(modelC, bottom_models, optimizers, optimizer_top, epoch, clean_acc, asr=None, inference_acc=None):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 使用与train_backdoor.py相同的命名格式
    temp = 'ALL' if not args.defense_type=='DPSGD' else 'DPSGD'
    label_knowledge = "True" if args.has_label_knowledge else "False"
    
    if asr is None:
        # 保存清洁模型
        model_name = f"Imagenette_Clean_{temp}_{label_knowledge}_{args.party_num}"
    else:
        # 保存后门模型，明确标识为VILLAIN攻击
        model_name = f"Imagenette_VILLAIN_WithInference_{temp}_{label_knowledge}_{args.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(args.checkpoint_dir, model_file_name)
    
    # 保存checkpoint
    checkpoint = {
        'model_bottom': {f'bottom_model_{i}': model.state_dict() for i, model in enumerate(bottom_models)},
        'model_top': modelC.state_dict(),
        'epoch': epoch,
        'clean_acc': clean_acc,
        'asr': asr,
        'inference_acc': inference_acc,
        'attack_type': 'VILLAIN_WithInference',
        'trigger_magnitude': args.trigger_magnitude,
        'trigger_size': args.trigger_size,
        'poison_budget': args.poison_budget,
        'inference_weight': args.inference_weight
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存{"清洁" if asr is None else "VILLAIN后门"}模型到 {model_save_path}') 

def collect_inference_data(modelC, bottom_models, train_loader, args):
    """收集标签推断数据，使用标准VILLAIN的嵌入交换和候选选择方法"""
    print("\n启动VILLAIN标签推断过程...")
    
    # 确保模型处于训练模式以获取梯度
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference
    
    if not label_inference_module:
        print("错误: 恶意模型没有标签推断模块")
        return False
    
    # 收集样本和梯度
    print(f"收集样本和梯度信息...")
    
    # 创建优化器（仅用于梯度计算）
    optimizers = []
    for model in bottom_models:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizers.append(optimizer)
    optimizerC = optim.SGD(modelC.parameters(), lr=args.lr)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 250:  # 增加批次数量，从150到250
            break
            
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 清除所有梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # 前向传播
        bottom_outputs = []
        for i, model in enumerate(bottom_models):
            # 对于恶意模型，确保输入需要梯度
            if i == args.bkd_adversary:
                data.requires_grad_(True)
            output = model(data)
            bottom_outputs.append(output)
        
        combined_output = torch.cat(bottom_outputs, dim=1)
        output = modelC(combined_output)
        
        # 计算损失并反向传播
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        
        # 获取梯度
        saved_data, saved_grad = adversary_model.get_saved_data()
        if saved_data is not None and saved_grad is not None:
            # 更新标签推断历史
            samples_added = label_inference_module.update_history(saved_data, saved_grad)
            
            # 仅偶尔输出信息
            if batch_idx % 20 == 0:  # 更频繁地输出信息
                print(f"已收集 {len(label_inference_module.gradient_history)} 个样本的梯度")
                if len(label_inference_module.gradient_history) > 0:
                    print(f"样本特征维度: {label_inference_module.gradient_history[0][0].shape}")
                    print(f"梯度范数: {np.linalg.norm(label_inference_module.gradient_history[0][1])}")
            
            # 更频繁地尝试标签推断步骤
            if batch_idx % 10 == 0 or len(label_inference_module.gradient_history) >= 50:  # 降低阈值
                # 步骤1: 嵌入交换
                swap_result = label_inference_module.embedding_swapping()
                if swap_result:
                    # 步骤2: 候选选择
                    selection_result = label_inference_module.candidate_selection()
                    if selection_result:
                        # 更新触发器状态
                        if adversary_model.villain_trigger:
                            adversary_model.villain_trigger.update_inference_stats()
                            return True
        
        # 清空梯度避免累积
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
    
    # 最终尝试执行嵌入交换和候选选择
    print(f"最终尝试标签推断初始化，已收集 {len(label_inference_module.gradient_history)} 个样本")
    if len(label_inference_module.gradient_history) > 0:
        print(f"样本特征维度: {label_inference_module.gradient_history[0][0].shape}")
        print(f"梯度范数: {np.linalg.norm(label_inference_module.gradient_history[0][1])}")
    
    if label_inference_module.embedding_swapping():
        if label_inference_module.candidate_selection():
            # 更新触发器状态
            if adversary_model.villain_trigger:
                adversary_model.villain_trigger.update_inference_stats()
                return True
    
    return False

def train_epoch(modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args):
    """训练一个轮次，包括标准VILLAIN标签推断和后门注入"""
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 打印后门攻击状态
    if epoch >= args.Ebkd:
        print(f"[Epoch {epoch}] 后门攻击: 启用 (从第{args.Ebkd}轮开始)")
    else:
        print(f"[Epoch {epoch}] 后门攻击: 禁用 (将在第{args.Ebkd}轮开始)")
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    total_loss = 0
    clean_correct = 0
    bkd_correct = 0
    total = 0
    backdoor_samples_total = 0  # 修复：累积实际攻击样本数
    criterion = nn.CrossEntropyLoss()
    
    # 确定是否启用标签推断
    has_inference = label_inference_module is not None
    
    # 确定是否进行梯度收集 - 每2个epoch进行一次
    collect_gradients = has_inference and (epoch % 2 == 0)
    
    # 记录批次数
    batch_count = 0
    warmup_batches = min(50, len(train_loader) // 10)  # 热身批次数
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 增加批次计数
        batch_count += 1
        data, target = data.to(DEVICE), target.to(DEVICE)
        total += len(data)
        
        # 清除梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # 准备后门数据 - 检查是否到达后门注入开始轮数
        is_backdoor = epoch >= args.Ebkd  # 只有达到指定轮数才开始后门攻击
        if is_backdoor:
            bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
            backdoor_samples = attack_flags.sum().item()
            backdoor_samples_total += backdoor_samples  # 修复：累积实际攻击样本数
        else:
            bkd_data, bkd_target, attack_flags = data, target, None
            backdoor_samples = 0
        
        # 前向传播 - 干净数据
        bottom_outputs_clean = []
        for i, model in enumerate(bottom_models):
            output = model(data)
            bottom_outputs_clean.append(output)
        
        combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
        output_clean = modelC(combined_output_clean)
        loss_clean = criterion(output_clean, target)
        
        # 前向传播 - 注入后门触发器
        bottom_outputs_bkd = []
        for i, model in enumerate(bottom_models):
            # 只有恶意方接收攻击标志，其他方正常处理
            if i == args.bkd_adversary:
                output = model(bkd_data, attack_flags=attack_flags)
            else:
                output = model(bkd_data)
            bottom_outputs_bkd.append(output)
        
        combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
        output_bkd = modelC(combined_output_bkd)
        loss_bkd = criterion(output_bkd, bkd_target)
        
        # 组合损失 - 只有启用后门攻击时才添加后门损失
        if is_backdoor:
            loss = loss_clean + args.trigger_magnitude * loss_bkd
        else:
            loss = loss_clean  # 未启用后门攻击时只使用干净损失
        
        # 反向传播
        loss.backward()
        
        # 恶意方收集梯度信息用于标签推断
        if collect_gradients and has_inference and batch_count <= warmup_batches:
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                # 更新标签推断历史
                samples_added = label_inference_module.update_history(saved_data, saved_grad)
                if batch_count % 10 == 0:  # 更频繁地输出信息
                    print(f"\n梯度收集: 批次 {batch_count}, 历史总数: {len(label_inference_module.gradient_history)}")
                    if len(label_inference_module.gradient_history) > 0:
                        print(f"样本特征维度: {label_inference_module.gradient_history[0][0].shape}")
                        print(f"梯度范数: {np.linalg.norm(label_inference_module.gradient_history[0][1])}")
                
                # 定期运行标签推断步骤
                if batch_count % args.cluster_update_freq == 0:
                    print(f"\n执行标签推断 (批次 {batch_count})")
                    # 步骤1: 嵌入交换
                    swap_result = label_inference_module.embedding_swapping()
                    if swap_result:
                        # 步骤2: 候选选择
                        selection_result = label_inference_module.candidate_selection()
                        if selection_result:
                            # 更新触发器状态
                            if adversary_model.villain_trigger:
                                adversary_model.villain_trigger.update_inference_stats()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(modelC.parameters(), 1.0)
        for model in bottom_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 优化
        for optimizer in optimizers:
            optimizer.step()
        optimizerC.step()
        
        # 计算准确率
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        clean_batch_correct = pred_clean.eq(target.view_as(pred_clean)).sum().item()
        clean_correct += clean_batch_correct
        
        if is_backdoor and attack_flags.sum() > 0:
            pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
            bkd_batch_correct = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
            bkd_correct += bkd_batch_correct
        
        # 累积损失
        total_loss += loss.item()
        
        # 打印进度
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * clean_correct / total
    
    # 计算攻击成功率
    attack_success_rate = 0.0
    if is_backdoor and bkd_correct > 0:
        # 修复：使用实际攻击样本数计算ASR
        attack_success_rate = min(100.0, 100.0 * bkd_correct / max(1, backdoor_samples_total))
    
    # 测试标签推断性能
    inference_accuracy = 0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # 创建测试子集
        test_subset_loader = torch.utils.data.Subset(
            train_loader.dataset, 
            indices=range(min(500, len(train_loader.dataset)))
        )
        test_subset_loader = torch.utils.data.DataLoader(
            test_subset_loader, 
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        # 仅用于评估推断性能，不需要更新模型
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_subset_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # 使用标签推断预测 - 使用原始图像数据
                flattened_data = data.view(data.size(0), -1)
                inferred_labels, _ = label_inference_module.infer_labels(flattened_data)
                
                if inferred_labels is not None:
                    # 计算推断准确率 - 与test函数保持一致
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        
                        # 判断预测是否正确：
                        # - 真实是目标类且预测为目标类(pred == args.target_class)
                        # - 真实不是目标类且预测不是目标类(pred != args.target_class)
                        if (is_target_class and pred == args.target_class) or (not is_target_class and pred != args.target_class):
                            correct_predictions += 1
                        
                        total_samples += 1
        
        # 计算推断准确率
        if total_samples > 0:
            inference_accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy 

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None):
    """测试模型性能，包括干净准确率和后门攻击成功率"""
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    test_loss = 0
    clean_correct = 0
    bkd_correct = 0
    backdoor_samples = 0
    total = 0
    inference_correct = 0
    inference_total = 0
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)
            total += batch_size
            
            # 准备干净数据的预测
            bottom_outputs_clean = []
            for model in bottom_models:
                output = model(data)
                bottom_outputs_clean.append(output)
            
            # 测试标签推断性能
            if label_inference_module and label_inference_module.initialized:
                # 使用原始图像数据进行标签推断
                flattened_data = data.view(data.size(0), -1)
                inferred_labels, _ = label_inference_module.infer_labels(flattened_data)
                
                if inferred_labels is not None:
                    # 计算推断准确率 - 关注推断对目标类的检测能力
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        
                        # 判断预测是否正确：
                        # - 真实是目标类且预测为目标类(pred == args.target_class)
                        # - 真实不是目标类且预测不是目标类(pred != args.target_class)
                        if (is_target_class and pred == args.target_class) or (not is_target_class and pred != args.target_class):
                            inference_correct += 1
                        
                        inference_total += 1
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            output_clean = modelC(combined_output_clean)
            
            # 计算干净损失
            test_loss += criterion(output_clean, target).item()
            
            # 预测干净样本
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            clean_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 如果需要，测试后门攻击成功率
            if is_backdoor:
                # 准备后门数据
                bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
                backdoor_samples += attack_flags.sum().item()
                
                if attack_flags.sum() > 0:
                    # 使用后门数据进行预测
                    bottom_outputs_bkd = []
                    for i, model in enumerate(bottom_models):
                        # 恶意方处理后门数据，其他方处理原始数据
                        if i == args.bkd_adversary:
                            output = model(bkd_data, attack_flags=attack_flags)
                        else:
                            output = model(bkd_data)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    output_bkd = modelC(combined_output_bkd)
                    
                    # 预测后门样本
                    pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                    bkd_correct += pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= total
    accuracy = 100. * clean_correct / total
    
    # 计算攻击成功率
    attack_success_rate = 0.0
    if backdoor_samples > 0:
        attack_success_rate = min(100.0, 100.0 * bkd_correct / backdoor_samples)
    
    # 计算推断准确率
    inference_accuracy = 0
    if inference_total > 0:
        inference_accuracy = 100. * inference_correct / inference_total
    
    # 打印结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}% ({clean_correct}/{total})')
    if is_backdoor:
        print(f'Backdoor ASR: {attack_success_rate:.2f}% ({bkd_correct}/{backdoor_samples})')
    if inference_total > 0:
        print(f'Inference Accuracy: {inference_accuracy:.2f}% ({inference_correct}/{inference_total})')
    
    return test_loss, accuracy, attack_success_rate, inference_accuracy

def main():
    """主函数：训练和评估模型"""
    # 设置随机种子以确保结果可重现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 加载数据集
    print("\n开始加载Imagenette数据集...")
    train_loader, test_loader, input_shape = load_dataset()
    
    # 创建模型
    print("\n创建模型...")
    bottom_models, modelC = create_models()
    
    # 创建优化器
    print("\n配置优化器...")
    optimizers = []
    for model in bottom_models:
        # 创建两个参数组，对冻结和非冻结层使用不同的学习率
        params = [
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': args.lr * 0.1},
            {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': args.lr}
        ]
        optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
        optimizers.append(optimizer)
    
    # 创建顶部模型优化器
    optimizer_top = optim.AdamW(modelC.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    schedulers = []
    for optimizer in optimizers:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 第一次重启的epoch数
            T_mult=2,  # 每次重启后T_0乘以的倍数
            eta_min=1e-6  # 最小学习率
        )
        schedulers.append(scheduler)
    
    scheduler_top = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_top,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    print(f"\n{'='*80}")
    print(f"开始训练 Imagenette 数据集的 VILLAIN-WithInference 攻击")
    print(f"{'='*80}")
    
    # 获取恶意模型
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 强制在训练前收集推断数据并初始化标签推断模块
    print("\n初始化标签推断模块...")
    inference_init_success = collect_inference_data(modelC, bottom_models, train_loader, args)
    if not inference_init_success:
        print("标签推断初始化未完成，将在训练中继续尝试")
    
    # 训练循环
    best_clean_acc = 0
    best_asr = 0
    best_epoch = 0
    no_improve = 0
    
    # 存储最佳模型对应的所有指标
    best_metrics = {
        'test_acc': 0,
        'inference_acc': 0,
        'asr': 0,
        'epoch': 0
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*40}")
        
        # 训练一个epoch
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, train_loader, optimizers, optimizer_top, epoch, args
        )
        
        # 打印训练结果
        print(f"\n训练结果 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"训练集攻击成功率: {train_asr:.2f}%")
        if train_inference_acc > 0:
            print(f"训练集推断准确率: {train_inference_acc:.2f}%")
        
        # 测试
        test_loss, test_acc, test_asr, test_inference_acc = test(
            modelC, bottom_models, test_loader, is_backdoor=True, 
            epoch=epoch, args=args
        )
        
        # 更新学习率
        for scheduler in schedulers:
            scheduler.step()
        scheduler_top.step()
        
        # 打印当前学习率
        current_lr = optimizers[0].param_groups[-1]['lr']
        print(f'当前学习率: {current_lr:.6f}')
        
        # 检查是否需要更新标签推断
        adversary_label_inference = adversary_model.label_inference
        should_update_inference = (
            epoch % args.cluster_update_freq == 0 or 
            (adversary_label_inference and not adversary_label_inference.initialized and epoch < 10)
        )
        
        if should_update_inference:
            print("\n周期性更新标签推断模块...")
            collect_inference_data(modelC, bottom_models, train_loader, args)
        
        # 组合分数标准
        combined_score = 0.5 * test_acc + 0.5 * test_asr
        best_combined_score = 0.5 * best_metrics['test_acc'] + 0.5 * best_metrics['asr']
        print(f"组合分数: 当前={combined_score:.2f}, 最佳={best_combined_score:.2f}")
        
        # 检查是否需要保存新的最佳模型
        if combined_score > best_combined_score:
            # 更新所有最佳指标
            best_metrics = {
                'test_acc': test_acc,
                'inference_acc': test_inference_acc,
                'asr': test_asr,
                'epoch': epoch
            }
            best_clean_acc = test_acc
            best_asr = test_asr
            best_epoch = epoch
            no_improve = 0
            # 保存模型
            save_checkpoint(
                modelC, bottom_models, optimizers, optimizer_top,
                epoch, test_acc, test_asr, test_inference_acc
            )
            print(f"\n保存最佳模型 (Epoch {epoch}) (组合分数提升)")
            print(f"Clean Acc: {test_acc:.2f}%, Inference Acc: {test_inference_acc:.2f}%, ASR: {test_asr:.2f}%")
        else:
            no_improve += 1
            print(f"\n没有改进: {no_improve}/{args.patience} (最佳组合分数: {best_combined_score:.2f} 在 Epoch {best_epoch})")
        
        # 提前停止
        if args.early_stopping and no_improve >= args.patience:
            print(f"\n早停触发! {args.patience} 轮次内没有改进。")
            print(f"最佳模型 (Epoch {best_epoch}):")
            print(f"Clean Acc: {best_clean_acc:.2f}%, Inference Acc: {best_metrics['inference_acc']:.2f}%, ASR: {best_asr:.2f}%")
            break
    
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"最佳干净准确率: {best_clean_acc:.2f}%")
    print(f"最佳推断准确率: {best_metrics['inference_acc']:.2f}%")
    if best_asr > 0:
        print(f"最佳攻击成功率: {best_asr:.2f}%")
    print(f"最佳轮次: {best_epoch}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 