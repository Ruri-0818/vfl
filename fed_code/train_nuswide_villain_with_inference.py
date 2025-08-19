import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
import time
import random
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA

# 扩展命令行参数
parser = argparse.ArgumentParser(description='针对NUS-WIDE数据集的VILLAIN攻击训练 (垂直联邦学习优化版)')
# VFL优化参数
parser.add_argument('--dataset', type=str, default='NUSWIDE', help='数据集名称 (NUSWIDE)')
parser.add_argument('--batch-size', type=int, default=64, help='训练批次大小 (VFL优化: 64)')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (VFL优化: 50)')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率 (VFL优化: 0.001，降低以提高稳定性)')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=float, default=0.5, help='VILLAIN触发器大小(VFL优化: 0.5，平衡clean acc和ASR)')
parser.add_argument('--trigger-magnitude', type=float, default=2.0, help='VILLAIN触发器强度(VFL优化: 2.0，平衡clean acc和ASR)')
parser.add_argument('--position', type=str, default='mid', help='恶意参与方位置')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别 (0=buildings)')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=3, help='参与方数量 (VFL优化: 3)')
parser.add_argument('--patience', type=int, default=10, help='早停轮数 (VFL优化: 10)')
parser.add_argument('--min-epochs', type=int, default=20, help='最小训练轮数 (VFL优化: 20)')
parser.add_argument('--max-epochs', type=int, default=100, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=1.0, help='后门损失权重 (VFL优化: 1.0，提高ASR)')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.2, help='毒化预算 (降低到0.2以提高clean acc)')
parser.add_argument('--Ebkd', type=int, default=5, help='后门注入开始轮数 (修正为5)')
parser.add_argument('--lr-multiplier', type=float, default=1.1, help='学习率倍增器 (VFL优化: 1.1)')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=5, help='类别数量 (固定为5个选择概念)')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='./data/NUS-WIDE', help='数据集目录')

# NUS-WIDE特定参数 (VFL优化)
parser.add_argument('--image-size', type=int, default=128, help='图像尺寸 (VFL优化: 128，降低复杂度)')
parser.add_argument('--aux-size', type=int, default=500, help='辅助数据大小 (VFL优化: 500)')
parser.add_argument('--proxy-model', type=str, default='mlp', help='代理模型类型')
parser.add_argument('--inference-interval', type=int, default=1, help='推断间隔')

# 标签推断相关参数 (VFL优化)
parser.add_argument('--inference-weight', type=float, default=0.05, help='标签推断损失权重 (降低到0.05)')
parser.add_argument('--history-size', type=int, default=1200, help='嵌入向量历史记录大小 (降低到1200)')
parser.add_argument('--cluster-update-freq', type=int, default=20, help='聚类更新频率 (降低到20批次)')
parser.add_argument('--inference-start-epoch', type=int, default=3, help='开始标签推断的轮数 (降低到3)')
parser.add_argument('--confidence-threshold', type=float, default=0.4, help='标签推断置信度阈值 (提高到0.4)')
parser.add_argument('--adaptive-threshold', action='store_true', help='是否使用自适应置信度阈值')
parser.add_argument('--feature-selection', action='store_true', help='是否启用特征选择以提高推断准确率')
parser.add_argument('--use-ensemble', action='store_true', help='是否使用集成方法提高推断准确率')

# 二元分类器参数 (VFL优化)
parser.add_argument('--gradient-mu', type=float, default=1.5, help='梯度范数阈值 (VFL优化: 1.5)')
parser.add_argument('--gradient-theta', type=float, default=1.2, help='梯度比率阈值 (VFL优化: 1.2)')
parser.add_argument('--binary-classifier', type=str, default='randomforest', choices=['randomforest', 'logistic'], 
                    help='二元分类器类型')

# VFL专用参数
parser.add_argument('--early-stopping', action='store_true', default=True, help='启用早停 (VFL默认启用)')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'], help='监控指标 (VFL优化：优先监控test_acc)')
parser.add_argument('--multi-label', action='store_true', default=False, help='使用多标签模式 (VFL默认关闭)')
parser.add_argument('--selected-concepts', type=str, nargs='+', 
                    default=['buildings', 'grass', 'animal', 'water', 'person'],
                    help='选择的5个NUS-WIDE概念 (固定使用这5个)')

# VFL特定优化参数
parser.add_argument('--vfl-mode', action='store_true', default=True, help='垂直联邦学习模式 (默认启用)')
parser.add_argument('--feature-split-strategy', type=str, default='random', choices=['random', 'semantic'], 
                    help='特征分割策略 (VFL)')
parser.add_argument('--communication-round', type=int, default=1, help='通信轮次 (VFL)')

# 新增参数以提高clean accuracy
parser.add_argument('--warmup-epochs', type=int, default=5, help='预热轮数，先训练clean model (增加到5)')
parser.add_argument('--clean-loss-weight', type=float, default=1.2, help='干净损失权重 (增加到1.2)')
parser.add_argument('--adaptive-loss-weight', action='store_true', default=True, help='自适应损失权重调整 (默认启用)')

# 新增参数以提高clean accuracy
parser.add_argument('--use-adam', action='store_true', default=False, help='使用Adam优化器而不是SGD')
parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['plateau', 'cosine', 'step'], help='学习率调度策略')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑参数')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# NUS-WIDE数据集类
class PreprocessedNUSWIDEDataset(Dataset):
    """预处理版本的NUS-WIDE数据集加载器 - 适配SciDB等平台的格式"""
    
    def __init__(self, data_dir, split='train', transform=None, num_classes=5, 
                 use_concepts=False, selected_concepts=None):
        """
        Args:
            data_dir: 数据集根目录 (包含images/, database_*.txt, test_*.txt等)
            split: 'train' 或 'test'
            transform: 图像变换
            num_classes: 类别数量
            use_concepts: 是否使用多标签模式
            selected_concepts: 选择的概念子集列表
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.use_concepts = use_concepts
        self.num_classes = num_classes
        
        # 设置选择的概念（用于标签映射）
        if selected_concepts is None:
            self.selected_concepts = ['buildings', 'grass', 'animal', 'water', 'person']
        else:
            self.selected_concepts = selected_concepts
        
        print(f"\n[预处理NUS-WIDE] 加载 {split} 数据集")
        print(f"数据目录: {data_dir}")
        print(f"标签模式: {'多标签' if use_concepts else '单标签'}")
        
        # 检查数据集路径
        self._check_dataset_structure()
        
        # 加载数据
        self._load_data()
        
        print(f"成功加载 {len(self.image_paths)} 个样本")
    
    def _check_dataset_structure(self):
        """检查预处理数据集的结构"""
        required_files = [
            'database_img.txt',
            'database_label.txt', 
            'test_img.txt',
            'test_label.txt',
            'images'
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"预处理NUS-WIDE数据集文件缺失: {missing_files}\n"
                f"请确保数据集目录 {self.data_dir} 包含所有必需文件"
            )
    
    def _load_data(self):
        """加载预处理格式的数据"""
        if self.split == 'train':
            img_file = os.path.join(self.data_dir, 'database_img.txt')
            label_file = os.path.join(self.data_dir, 'database_label.txt')
        else:
            img_file = os.path.join(self.data_dir, 'test_img.txt')
            label_file = os.path.join(self.data_dir, 'test_label.txt')
        
        # 读取图像路径
        with open(img_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # 读取标签
        with open(label_file, 'r') as f:
            label_lines = [line.strip() for line in f.readlines()]
        
        # 处理标签格式
        self.labels = []
        all_labels = set()
        
        for line in label_lines:
            if line:
                # 解析多标签格式（空格分隔的数字）
                multi_label = [int(x) for x in line.split() if x.isdigit()]
                # 确保标签在有效范围内
                multi_label = [x % self.num_classes for x in multi_label]
                all_labels.update(multi_label)
                self.labels.append(multi_label)
            else:
                self.labels.append([0])  # 没有标签的样本分配到类别0
        
        # 分析标签分布
        print(f"发现的所有标签: {sorted(all_labels)}")
        print(f"标签范围: {min(all_labels) if all_labels else 0} - {max(all_labels) if all_labels else 0}")
        
        # 更新类别数
        if all_labels:
            self.num_classes = max(all_labels) + 1  # 标签从0开始
        
        # 如果需要单标签模式，转换多标签为单标签
        if not self.use_concepts:
            self._convert_to_single_label()
        
        # 限制数据量以提高训练效率（VFL优化）
        max_samples = 3000 if self.split == 'train' else 600
        if len(self.image_paths) > max_samples:
            print(f"限制样本数量从 {len(self.image_paths)} 到 {max_samples}")
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]
        
        # 转换相对路径为绝对路径
        self.image_paths = [os.path.join(self.data_dir, path) for path in self.image_paths]
        
        # 验证图像文件存在
        valid_indices = []
        for i, img_path in enumerate(self.image_paths):
            if os.path.exists(img_path):
                valid_indices.append(i)
            elif i < 10:  # 只输出前10个错误
                print(f"警告: 图像文件不存在: {img_path}")
        
        # 过滤有效样本
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        
        print(f"有效样本数: {len(self.image_paths)}")
    
    def _convert_to_single_label(self):
        """将多标签转换为单标签格式"""
        print("转换多标签为单标签格式...")
        
        single_labels = []
        for multi_label in self.labels:
            if multi_label:
                # 使用第一个标签作为主标签
                single_label = multi_label[0]
                # 确保标签在有效范围内 [0, num_classes-1]
                single_label = single_label % self.num_classes
            else:
                # 没有标签的样本分配到类别0
                single_label = 0
            single_labels.append(single_label)
        
        self.labels = single_labels
        
        # 验证所有标签都在有效范围内
        invalid_labels = [label for label in self.labels if label < 0 or label >= self.num_classes]
        if invalid_labels:
            print(f"警告: 发现无效标签，将重新映射")
            # 重新映射所有标签到有效范围
            self.labels = [max(0, min(label, self.num_classes-1)) for label in self.labels]
        
        # 统计标签分布
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"单标签分布: {dict(sorted(label_counts.items()))}")
        print(f"标签范围验证: 最小={min(self.labels)}, 最大={max(self.labels)}, 类别数={self.num_classes}")
        
        # 确保没有超出范围的标签
        assert all(0 <= label < self.num_classes for label in self.labels), \
            f"标签超出范围! 标签范围: [{min(self.labels)}, {max(self.labels)}], 期望范围: [0, {self.num_classes-1}]"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果图像加载失败，返回一个黑色图像
            print(f"警告: 无法加载图像 {img_path}: {str(e)}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        target = self.labels[idx]
        
        # 确保标签在有效范围内
        if isinstance(target, list):
            if target:
                target = target[0] % self.num_classes
            else:
                target = 0
        else:
            target = target % self.num_classes
        
        # 如果是多标签，转换为tensor
        if self.use_concepts:
            # 创建one-hot向量
            one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            one_hot[target] = 1.0
            target = one_hot
        
        return img, target

# 为了向后兼容，创建一个别名
NUSWIDEDataset = PreprocessedNUSWIDEDataset

# 标签推断器实现（与CIFAR版本相同，但调整特征维度）
class LabelInferenceModule:
    """增强版标签推断模块，使用改进的分类器提高推断准确率"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.history_size = args.history_size
        self.confidence_threshold = args.confidence_threshold
        
        # 存储每个类别的数据
        self.data_by_class = {c: [] for c in range(num_classes)}
        self.labels_by_class = {c: [] for c in range(num_classes)}
        
        # 主分类器和备份分类器
        self.classifier = None
        self.backup_classifier = None
        self.ensemble_weights = [0.7, 0.3]  # 主分类器权重更高
        
        self.means = None
        self.initialized = False
        self.min_samples_per_class = 20  # 每个类别的最小样本数
        self.update_counter = 0  # 追踪更新次数
        
        # 设置所需的最小样本数
        self.min_samples = max(100, 10 * num_classes)
        
        # 特征选择
        self.selected_features = None
        self.feature_importance = None
        
        # 性能指标跟踪
        self.accuracy_history = []
        self.confidence_history = []
        
        # 标记是否使用梯度特征
        self.use_gradient_features = False
        
        # 存储训练时的特征维度
        self.training_feature_dim = None
        
        print(f"增强版标签推断模块创建完成: 特征维度={feature_dim}, 类别数={num_classes}, "
              f"历史大小={self.history_size}, 置信度阈值={self.confidence_threshold}")
    
    def _process_features(self, features):
        """统一处理特征，确保训练和推断时使用相同的特征处理方式"""
        features_np = features.detach().cpu().numpy()
        
        # 打印原始特征维度
        print(f"原始特征维度: {features_np.shape}")
        
        # 如果是第一次处理特征，记录维度
        if self.training_feature_dim is None:
            # 初始特征维度应该是49152，加上梯度特征后是49153
            self.training_feature_dim = features_np.shape[1] + 1
            print(f"记录初始特征维度: {self.training_feature_dim}")
        
        # 添加梯度特征维度
        if features_np.shape[1] == self.training_feature_dim - 1:
            # 添加一个零向量作为梯度特征
            gradient_norms = np.zeros((features_np.shape[0], 1))
            features_np = np.column_stack((features_np, gradient_norms))
            print(f"添加梯度特征维度后: {features_np.shape}")
        elif features_np.shape[1] != self.training_feature_dim:
            raise ValueError(f"特征维度不匹配: 输入特征维度 {features_np.shape[1]} 与训练特征维度 {self.training_feature_dim} 不一致")
        
        return features_np
    
    def update_history(self, features, labels):
        """更新历史数据，保持每个类别的样本平衡"""
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 打印原始特征维度
        print(f"原始特征维度: {features_np.shape}")
        
        # 如果是第一次处理特征，记录维度
        if self.training_feature_dim is None:
            # 初始特征维度应该是49152，加上梯度特征后是49153
            self.training_feature_dim = features_np.shape[1] + 1
            print(f"记录初始特征维度: {self.training_feature_dim}")
            
        # 如果特征维度不匹配，直接报错
        if features_np.shape[1] != self.training_feature_dim - 1:
            raise ValueError(f"特征维度不匹配: 输入特征维度 {features_np.shape[1]} 与训练特征维度 {self.training_feature_dim} 不一致")
        
        # 处理梯度数据 - 使用梯度的范数作为特征
        if len(labels_np.shape) > 1:  # 如果是梯度数据
            # 计算每个样本的梯度范数
            gradient_norms = np.linalg.norm(labels_np, axis=1)
            # 使用梯度范数作为特征
            features_np = np.column_stack((features_np, gradient_norms))
            self.use_gradient_features = True
            print(f"使用梯度特征，特征维度: {features_np.shape[1]}")
            
            # 根据梯度范数确定标签
            # 使用梯度范数的中位数作为阈值
            median_norm = np.median(gradient_norms)
            labels_np = (gradient_norms > median_norm).astype(int)
        
        # 平衡系数: 对样本数少的类别更有可能接受新样本
        added_samples = 0
        class_counts = {c: len(self.data_by_class[c]) for c in range(self.num_classes)}
        max_samples_per_class = self.history_size // self.num_classes
        
        for feature, label in zip(features_np, labels_np):
            label_int = int(label)
            
            # 跳过无效标签
            if label_int < 0 or label_int >= self.num_classes:
                continue
                
            # 确保类别存在于字典中
            if label_int not in self.data_by_class:
                self.data_by_class[label_int] = []
                self.labels_by_class[label_int] = []
        
            # 计算当前类别的样本数 
            current_count = class_counts.get(label_int, 0)
            
            # 计算接受率 - 样本越少，接受率越高
            if current_count < max_samples_per_class:
                # 如果样本数不足，直接添加
                self.data_by_class[label_int].append(feature)
                self.labels_by_class[label_int].append(label_int)
                added_samples += 1
                class_counts[label_int] = current_count + 1
        
        return added_samples
    
    def get_total_samples(self):
        """获取总样本数"""
        return sum(len(samples) for samples in self.data_by_class.values())
    
    def get_samples_per_class(self):
        """获取每个类别的样本数"""
        return {c: len(samples) for c, samples in self.data_by_class.items()}
    
    def update_class_stats(self, force=False):
        """更新类别统计信息和分类器"""
        # 检查是否有足够的样本
        if self.get_total_samples() < self.min_samples and not force:
            print(f"样本不足，无法更新类别统计信息 ({self.get_total_samples()}/{self.min_samples})")
            return False
        
        # 准备训练数据
        X_train = []
        y_train = []
        valid_classes = []
        
        # 收集每个类别的样本
        for c in range(self.num_classes):
            if len(self.data_by_class[c]) >= self.min_samples_per_class:
                valid_classes.append(c)
                X_train.extend(self.data_by_class[c])
                y_train.extend(self.labels_by_class[c])
        
        if len(valid_classes) < 2:
            print(f"有效类别数不足: {len(valid_classes)}")
            return False
        
        # 转换为numpy数组
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # 训练分类器
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # 保存旧分类器作为备份
            if self.classifier is not None:
                self.backup_classifier = self.classifier
            
            # 使用更强大的随机森林配置
            rf_classifier = RandomForestClassifier(
                n_estimators=300,  # 增加树的数量
                max_depth=20,      # 增加深度
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
            
            # 训练主分类器
            rf_classifier.fit(X_train, y_train)
            
            # 训练备份分类器（逻辑回归）
            lr_classifier = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
            lr_classifier.fit(X_train, y_train)
            
            # 选择性能更好的分类器作为主分类器
            rf_score = rf_classifier.score(X_train, y_train)
            lr_score = lr_classifier.score(X_train, y_train)
            
            if rf_score >= lr_score:
                self.classifier = rf_classifier
                self.backup_classifier = lr_classifier
                print(f"选择随机森林作为主分类器 (RF: {rf_score*100:.2f}% vs LR: {lr_score*100:.2f}%)")
            else:
                self.classifier = lr_classifier
                self.backup_classifier = rf_classifier
                print(f"选择逻辑回归作为主分类器 (LR: {lr_score*100:.2f}% vs RF: {rf_score*100:.2f}%)")
            
            # 计算训练集准确率
            train_accuracy = self.classifier.score(X_train, y_train)
            print(f"分类器训练完成，训练集准确率: {train_accuracy*100:.2f}%")
            
            # 记录准确率历史
            self.accuracy_history.append(train_accuracy * 100)
            
            # 不使用特征选择，保持原始特征维度
            self.selected_features = None
            self.feature_importance = None
            print(f"使用完整特征集 ({X_train.shape[1]} 维)")
            
            self.initialized = True
            print(f"类别统计信息更新成功: {len(valid_classes)}/{self.num_classes} 类别有效")
            return True
        except Exception as e:
            print(f"分类器训练失败: {str(e)}")
            self.classifier = None
            self.initialized = False
            return False
    
    def infer_labels(self, features):
        """使用训练好的分类器推断标签"""
        if not self.initialized or self.classifier is None:
            return None, None
        
        # 使用统一的特征处理
        features_np = self._process_features(features)
        
        # 确保特征维度完全匹配
        if features_np.shape[1] != self.training_feature_dim:
            print(f"警告: 特征维度不匹配，进行调整")
            print(f"输入特征维度: {features_np.shape[1]}, 期望维度: {self.training_feature_dim}")
            
            if features_np.shape[1] < self.training_feature_dim:
                # 如果维度不足，用零填充
                padding = np.zeros((features_np.shape[0], self.training_feature_dim - features_np.shape[1]))
                features_np = np.column_stack((features_np, padding))
                print(f"添加零填充后维度: {features_np.shape}")
            else:
                # 如果维度过多，截断
                features_np = features_np[:, :self.training_feature_dim]
                print(f"截断后维度: {features_np.shape}")
        
        # 检查特征维度是否匹配
        if self.selected_features is not None:
            if len(self.selected_features) != features_np.shape[1]:
                raise ValueError(f"特征选择器维度不匹配: 特征维度 {features_np.shape[1]} 与选择器维度 {len(self.selected_features)} 不一致")
            
            # 确保特征维度匹配后再进行选择
            features_np = features_np[:, self.selected_features]
        
        # 预测类别和概率
        try:
            # 主分类器预测
            pred_probs = self.classifier.predict_proba(features_np)
            
            # 如果备份分类器可用，使用集成预测
            if self.backup_classifier is not None:
                try:
                    backup_probs = self.backup_classifier.predict_proba(features_np)
                    pred_probs = self.ensemble_weights[0] * pred_probs + self.ensemble_weights[1] * backup_probs
                except Exception as e:
                    print(f"备份分类器预测失败: {str(e)}")
            
            pred_labels = np.argmax(pred_probs, axis=1)
            confidence = np.max(pred_probs, axis=1)
            
            # 对置信度低的预测应用阈值
            for i in range(len(pred_labels)):
                if confidence[i] < self.confidence_threshold:
                    pred_labels[i] = -1  # 置信度不足，标记为未知
            
            # 更新置信度历史
            if len(confidence) > 0:
                self.confidence_history.append(np.mean(confidence))
            
            return pred_labels.tolist(), confidence.tolist()
        except Exception as e:
            raise RuntimeError(f"标签推断失败: {str(e)}")

    def get_metrics(self):
        """返回当前推断模块的性能指标"""
        metrics = {
            "initialized": self.initialized,
            "total_samples": self.get_total_samples(),
            "samples_per_class": self.get_samples_per_class(),
            "average_confidence": np.mean(self.confidence_history[-10:]) if len(self.confidence_history) >= 10 else 0,
            "average_accuracy": np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else 0
        }
        return metrics

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
        
        # 确定要修改的特征数量 - 增加触发器大小
        num_features = int(embed_dim * self.trigger_size * 1.2)  # 增加20%
        
        # 创建触发器掩码
        trigger_mask = torch.zeros_like(embeddings)
        
        # 如果还没有特征索引，或者标签推断模块未初始化
        if self.feature_indices is None or not self.is_initialized:
            # 使用更智能的特征选择
            if embed_dim > 1000:
                # 对于高维特征，选择中间部分
                start_idx = (embed_dim - num_features) // 2
                self.feature_indices = torch.arange(start_idx, start_idx + num_features, device=device)
            else:
                # 对于低维特征，选择所有特征
                self.feature_indices = torch.arange(num_features, device=device)
            
            # 仅在特征索引首次创建时输出信息
            if self.batch_count % 50 == 0:
                print(f"使用智能特征选择创建触发器 (强度={self.trigger_magnitude:.2f})")
        
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
                    # 使用更强的触发器
                    trigger_mask[i, self.feature_indices] = self.trigger_magnitude * 1.2  # 增加20%强度
                    
                    # 添加一些随机性以增加鲁棒性
                    if random.random() < 0.3:  # 30%的概率添加额外扰动
                        noise = torch.randn_like(trigger_mask[i, self.feature_indices]) * 0.1
                        trigger_mask[i, self.feature_indices] += noise
        else:
            # 如果没有有效的推断标签，使用更保守的策略
            if random.random() < 0.1:  # 降低到10%的概率
                # 随机应用触发器，但使用更小的强度
                random_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                for i in range(batch_size):
                    if random.random() < 0.2:  # 降低到20%的样本
                        random_labels[i] = 1
                        trigger_mask[i, self.feature_indices] = self.trigger_magnitude * 0.8  # 降低20%强度
            
        self.batch_count += 1
        return trigger_mask

# ResNet基本块
class ResBlock(nn.Module):
    """ResNet基本块"""
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
    
    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 快捷路径
        shortcut = self.shortcut(x)
        
        # 合并主路径和快捷路径
        out += shortcut
        out = self.relu(out)
        
        return out

class NUSWIDEBottomModel(nn.Module):
    """NUS-WIDE底部模型，简化版本以提高clean accuracy"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(NUSWIDEBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 简化的特征提取器 - 降低复杂度以提高clean accuracy
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 简化的分类器
        feature_size = 256 * 4 * 4  # 4096
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 降低dropout率
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # 用于存储当前批次的数据和梯度
        self.current_batch_data = None
        self.current_batch_grad = None
        
        # 初始化权重
        self._initialize_weights()
        
        # 如果是恶意模型，初始化标签推断模块
        if is_adversary and args is not None:
            self.villain_trigger = None
            self.label_inference = None  # 稍后初始化
            print(f"创建恶意底部模型 (ID={args.bkd_adversary}) - 简化版本")
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_villain_trigger(self, villain_trigger):
        """设置VILLAIN触发器"""
        if self.is_adversary:
            self.villain_trigger = villain_trigger
            # 创建标签推断模块
            if not hasattr(self, 'label_inference') or self.label_inference is None:
                print("为恶意模型创建标签推断模块")
                self.label_inference = LabelInferenceModule(
                    feature_dim=self.output_dim,
                    num_classes=self.args.num_classes,
                    args=self.args
                )
                # 将标签推断模块传递给触发器
                self.villain_trigger.set_label_inference(self.label_inference)
            else:
                # 确保触发器有对标签推断模块的引用
                self.villain_trigger.set_label_inference(self.label_inference)
            
            # 如果标签推断模块已初始化，立即更新触发器状态
            if self.label_inference.initialized:
                print("标签推断模块已初始化，正在更新触发器状态...")
                self.villain_trigger.update_inference_stats()
    
    def forward(self, x, attack_flags=None):
        """前向传播，包括恶意触发器注入和梯度收集"""
        # 如果是恶意模型，保存输入数据用于梯度收集
        if self.is_adversary and self.training:
            self.current_batch_data = x.detach()
            x.requires_grad_(True)
        
        # 简化的特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        
        # 展平
        x_flat = x.view(x.size(0), -1)
        
        # 分类
        feat = self.classifier(x_flat)
        
        # 如果是恶意模型并且设置了攻击标志，注入触发器
        if self.is_adversary and attack_flags is not None and self.villain_trigger is not None:
            batch_size = feat.size(0)
            
            # 使用标签推断模块
            if hasattr(self, 'label_inference') and self.label_inference is not None and self.label_inference.initialized:
                # 推断样本标签 - 使用原始特征数据，适配128x128图像尺寸
                original_features = self.current_batch_data.view(self.current_batch_data.size(0), -1)
                inferred_labels, _ = self.label_inference.infer_labels(original_features)
                
                if inferred_labels is not None:
                    # 将标签转换为触发器格式：1表示非目标类(需要触发)，0表示目标类(不需要触发)
                    torch_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    
                    # 遍历推断的标签
                    for i, label in enumerate(inferred_labels):
                        if attack_flags[i] and label == 1:
                            # 只有标签推断为非目标类（label=1），且被标记为攻击样本的才应用触发器
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
                if random.random() < 0.2:  # 只有20%的概率随机应用触发器
                    # 随机应用触发器
                    random_labels = torch.zeros(batch_size, dtype=torch.long, device=feat.device)
                    for i in range(batch_size):
                        if attack_flags[i] and random.random() < 0.3:  # 30%的样本被触发
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

class NUSWIDETopModel(nn.Module):
    """NUS-WIDE top model - VFL优化版本"""
    def __init__(self, input_dim=256, num_classes=5):
        super(NUSWIDETopModel, self).__init__()
        # VFL优化：简化网络结构，提高训练效率
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
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

def load_dataset(dataset_name, data_dir, batch_size, image_size=224):
    """加载数据集 - VFL优化版本，支持预处理的NUS-WIDE数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集 (VFL优化版)")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    
    # VFL优化：简化数据增强，提高训练效率
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n2. 检查NUS-WIDE数据集路径...")
    
    if dataset_name.upper() == 'NUSWIDE':
        # 使用预处理的NUS-WIDE数据集
        # 检查是否有嵌套的NUS-WIDE目录
        nuswide_subdir = os.path.join(data_dir, 'NUS-WIDE')
        if os.path.exists(nuswide_subdir) and os.path.exists(os.path.join(nuswide_subdir, 'database_img.txt')):
            actual_data_dir = nuswide_subdir
            print(f"检测到嵌套目录，使用: {actual_data_dir}")
        else:
            actual_data_dir = data_dir
            print(f"使用数据目录: {actual_data_dir}")
        
        # 获取选择的概念 (VFL优化：固定使用5个概念)
        selected_concepts = getattr(args, 'selected_concepts', ['buildings', 'grass', 'animal', 'water', 'person'])
        print(f"[VFL优化] 选择的概念: {selected_concepts}")
        
        # VFL优化：默认使用单标签模式
        use_multi_label = hasattr(args, 'multi_label') and args.multi_label
        num_classes = 5  # 固定使用5个类别
        
        print("\n3. 加载预处理的NUS-WIDE数据集...")
        print("正在加载NUS-WIDE train 数据...")
        train_dataset = NUSWIDEDataset(
            data_dir=actual_data_dir, 
            split='train', 
            transform=transform_train,
            num_classes=num_classes,
            use_concepts=use_multi_label,
            selected_concepts=selected_concepts
        )
        
        print("正在加载NUS-WIDE test 数据...")
        test_dataset = NUSWIDEDataset(
            data_dir=actual_data_dir, 
            split='test', 
            transform=transform_test,
            num_classes=num_classes,
            use_concepts=use_multi_label,
            selected_concepts=selected_concepts
        )
        
        # 更新全局类别数
        args.num_classes = num_classes
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print("\n4. 创建数据加载器...")
    
    # VFL优化：减少num_workers以降低资源消耗
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print("\n[VFL优化] 数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"选择的概念: {selected_concepts}")
    print(f"类别数量: {num_classes}")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"标签类型: {'多标签' if use_multi_label else '单标签分类 (VFL默认)'}")
    print(f"参与方数量: {args.party_num}")
    print(f"数据目录: {actual_data_dir}")
    
    print(f"\n{'='*50}")
    print("预处理NUS-WIDE数据集加载完成！(VFL优化版)")
    print(f"{'='*50}\n")
    
    return train_loader, test_loader, (3, image_size, image_size)

def create_models():
    """创建模型"""
    output_dim = 64  # 每个底部模型的输出维度
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        if i == args.bkd_adversary:
            # 创建恶意模型
            model = NUSWIDEBottomModel(
                input_dim=3,  # RGB图像
                output_dim=output_dim,
                is_adversary=True,
                args=args
            )
        else:
            # 创建正常模型
            model = NUSWIDEBottomModel(
                input_dim=3,
                output_dim=output_dim
            )
        model = model.to(DEVICE)  # 确保模型在GPU上
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = NUSWIDETopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)  # 确保模型在GPU上
    
    # 创建并设置VILLAIN触发器
    villain_trigger = VILLAINTrigger(args)
    bottom_models[args.bkd_adversary].set_villain_trigger(villain_trigger)
    
    return bottom_models, modelC

def prepare_backdoor_data(data, target):
    """准备后门数据，注入后门触发器"""
    batch_size = data.size(0)
    
    # 计算毒化样本数量，基于毒化预算
    attack_portion = int(batch_size * args.poison_budget)
    
    # 设置攻击标志 
    attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    if attack_portion > 0:
        attack_flags[:attack_portion] = True
    
    # 修改标签为目标类别
    bkd_target = target.clone()
    bkd_target[attack_flags] = args.target_class
    
    return data, bkd_target, attack_flags

def train_epoch(modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args):
    """训练一个轮次，包括标准VILLAIN标签推断和后门注入 - 平衡版本"""
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    total_loss = 0
    clean_correct = 0
    bkd_correct = 0
    total = 0
    backdoor_samples_total = 0  # 修复：累积实际攻击样本数
    
    # 使用标签平滑的交叉熵损失
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"使用标签平滑: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 确定是否启用标签推断
    has_inference = label_inference_module is not None
    
    # 确定是否进行梯度收集 - 在Ebkd之前积极收集
    collect_gradients = has_inference and epoch <= args.Ebkd + 5  # 在后门开始后再收集5个epoch
    
    # 确定是否是预热阶段 - 预热阶段应该在Ebkd之前
    is_warmup = epoch < args.Ebkd  # 严格在Ebkd之前才是预热
    
    # 确定是否启动后门攻击 - 只有在Ebkd之后且不在预热阶段
    enable_backdoor = epoch >= args.Ebkd and not is_warmup
    
    # 更保守的损失权重调整策略，优先保证clean accuracy
    if hasattr(args, 'adaptive_loss_weight') and args.adaptive_loss_weight:
        if epoch < args.Ebkd:
            # 预热阶段：完全专注于clean performance
            backdoor_weight = 0.0
            clean_weight = 1.0
        elif epoch < args.Ebkd + 5:
            # 后门启动初期：逐渐引入后门，但仍优先clean acc
            backdoor_weight = args.backdoor_weight * 0.2 * (epoch - args.Ebkd + 1) / 5
            clean_weight = 1.0
        else:
            # 后期：平衡两者，但仍然给clean更高权重
            backdoor_weight = args.backdoor_weight * 0.6  # 永远不超过60%权重
            clean_weight = 1.0
    else:
        if epoch < args.Ebkd:
            backdoor_weight = 0.0
            clean_weight = 1.0
        else:
            backdoor_weight = args.backdoor_weight * 0.5  # 减半后门权重
            clean_weight = getattr(args, 'clean_loss_weight', 1.0)
    
    # 记录批次数
    batch_count = 0
    warmup_batches = min(100, len(train_loader) // 6)  # 减少热身批次
    
    print(f"Epoch {epoch}: 预热阶段={is_warmup}, 启用后门={enable_backdoor}, 后门权重={backdoor_weight:.3f}, clean权重={clean_weight:.3f}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 增加批次计数
        batch_count += 1
        data, target = data.to(DEVICE), target.to(DEVICE)
        total += len(data)
        
        # 清除梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # 前向传播 - 干净数据
        bottom_outputs_clean = []
        for i, model in enumerate(bottom_models):
            output = model(data)
            bottom_outputs_clean.append(output)
        
        combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
        output_clean = modelC(combined_output_clean)
        loss_clean = criterion(output_clean, target)
        
        # 后门攻击处理
        loss_backdoor = 0
        backdoor_samples = 0
        if enable_backdoor:
            # 准备后门数据
            bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
            backdoor_samples = attack_flags.sum().item()
            backdoor_samples_total += backdoor_samples  # 修复：累积实际攻击样本数
            
            if backdoor_samples > 0:
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
                loss_backdoor = criterion(output_bkd, bkd_target)
        
        # 组合损失 - 平衡clean accuracy和backdoor effectiveness
        if enable_backdoor and backdoor_samples > 0:
            loss = clean_weight * loss_clean + backdoor_weight * loss_backdoor
        else:
            loss = clean_weight * loss_clean
        
        # 反向传播
        loss.backward()
        
        # 恶意方收集梯度信息用于标签推断 - 优化版本
        if collect_gradients and has_inference:
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                # 更新标签推断历史
                original_data = saved_data.view(saved_data.size(0), -1)
                samples_added = label_inference_module.update_history(original_data, saved_grad)
                
                # 更频繁的调试信息
                if batch_count % 10 == 0:
                    total_samples = label_inference_module.get_total_samples()
                    print(f"梯度收集: 批次{batch_count}, 历史样本数:{total_samples}, 初始化状态:{label_inference_module.initialized}")
                
                # 尝试更新类别统计信息
                if not label_inference_module.initialized and total_samples >= label_inference_module.min_samples:
                    print(f"尝试更新类别统计信息 (批次{batch_count})")
                    if label_inference_module.update_class_stats():
                        print(f"标签推断在批次{batch_count}初始化成功！")
                        # 更新触发器状态
                        if adversary_model.villain_trigger:
                            adversary_model.villain_trigger.is_initialized = True
        
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(modelC.parameters(), args.grad_clip)
        for model in bottom_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # 优化
        for optimizer in optimizers:
            optimizer.step()
        optimizerC.step()
        
        # 计算准确率
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        clean_batch_correct = pred_clean.eq(target.view_as(pred_clean)).sum().item()
        clean_correct += clean_batch_correct
        
        if enable_backdoor and backdoor_samples > 0:
            pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
            bkd_batch_correct = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
            bkd_correct += bkd_batch_correct
        
        # 累积损失
        total_loss += loss.item()
        
        # 打印进度
        if batch_idx % (args.log_interval * 2) == 0:
            current_clean_acc = 100. * clean_correct / total
            if enable_backdoor and bkd_correct > 0:
                # 修复：使用实际攻击样本数计算ASR
                current_asr = 100. * bkd_correct / max(1, backdoor_samples_total)
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]'
                      f'\tLoss: {loss.item():.6f}, Clean Acc: {current_clean_acc:.2f}%, ASR: {current_asr:.2f}%')
            else:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]'
                      f'\tLoss: {loss.item():.6f}, Clean Acc: {current_clean_acc:.2f}%')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * clean_correct / total
    
    # 计算攻击成功率
    attack_success_rate = 0.0
    if enable_backdoor and bkd_correct > 0:
        # 修复：使用实际攻击样本数计算ASR
        attack_success_rate = min(100.0, 100.0 * bkd_correct / max(1, backdoor_samples_total))
    
    # 更全面的推断准确率计算
    inference_accuracy = 0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # 使用一个小的验证集测试推断性能
        test_subset_size = min(200, len(train_loader.dataset) // 10)
        test_indices = list(range(test_subset_size))
        test_subset = torch.utils.data.Subset(train_loader.dataset, test_indices)
        test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)
        
        correct_predictions = 0
        total_samples = 0
        target_class_count = 0
        non_target_class_count = 0
        
        print("\n开始计算推断准确率...")
        with torch.no_grad():
            for data, target in test_subset_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # 使用标签推断预测
                original_data = data.view(data.size(0), -1)
                inferred_labels, confidences = label_inference_module.infer_labels(original_data)
                
                if inferred_labels is not None:
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        if is_target_class:
                            target_class_count += 1
                        else:
                            non_target_class_count += 1
                        
                        # 输出详细的预测信息
                        if j < 5:  # 只打印前5个样本的详细信息
                            print(f"样本 {j}: 真实标签={true}, 目标类={is_target_class}, 预测={pred}, 置信度={confidences[j]:.3f}")
                        
                        # VILLAIN推断逻辑：pred=0表示目标类，pred=1表示非目标类
                        if (is_target_class and pred == 0) or (not is_target_class and pred == 1):
                            correct_predictions += 1
                        total_samples += 1
        
        if total_samples > 0:
            inference_accuracy = 100.0 * correct_predictions / total_samples
            print(f"\n推断准确率统计:")
            print(f"总样本数: {total_samples}")
            print(f"目标类样本数: {target_class_count}")
            print(f"非目标类样本数: {non_target_class_count}")
            print(f"正确预测数: {correct_predictions}")
            print(f"推断准确率: {inference_accuracy:.2f}%")
            
            # 输出分类器的性能指标
            if hasattr(label_inference_module, 'get_metrics'):
                metrics = label_inference_module.get_metrics()
                print("\n分类器性能指标:")
                print(f"平均置信度: {metrics['average_confidence']:.3f}")
                print(f"平均准确率: {metrics['average_accuracy']:.2f}%")
                print(f"每个类别的样本数: {metrics['samples_per_class']}")
    
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
                # 直接使用原始图像数据
                original_data = data.view(data.size(0), -1)
                inferred_labels, _ = label_inference_module.infer_labels(original_data)
                
                if inferred_labels is not None:
                    # 计算推断准确率 - 关注推断对目标类的检测能力
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        
                        # VILLAIN推断逻辑：pred=0表示目标类，pred=1表示非目标类
                        # 判断预测是否正确：
                        # - 真实是目标类且预测为0（目标类）
                        # - 真实不是目标类且预测为1（非目标类）
                        if (is_target_class and pred == 0) or (not is_target_class and pred == 1):
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

def save_checkpoint(modelC, bottom_models, optimizers, optimizer_top, epoch, clean_acc, asr=None, inference_acc=None):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    temp = 'ALL' if not args.defense_type=='DPSGD' else 'DPSGD'
    label_knowledge = "True" if args.has_label_knowledge else "False"
    
    if asr is None:
        model_name = f"{args.dataset}_Clean_{temp}_{label_knowledge}_{args.party_num}"
    else:
        model_name = f"{args.dataset}_VILLAIN_WithInference_{temp}_{label_knowledge}_{args.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(args.checkpoint_dir, model_file_name)
    
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
    print(f'保存模型到 {model_save_path}')

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
            # 更新标签推断历史 - 直接使用原始图像数据
            original_data = saved_data.view(saved_data.size(0), -1)
            samples_added = label_inference_module.update_history(original_data, saved_grad)
            
            # 仅偶尔输出信息
            if batch_idx % 20 == 0:  # 更频繁地输出信息
                total_samples = label_inference_module.get_total_samples()
                print(f"已收集 {total_samples} 个样本")
                if total_samples > 0:
                    print(f"样本特征维度: {original_data.shape[1]}")
                    print(f"梯度范数: {torch.norm(saved_grad).item():.4f}")
            
            # 更频繁地尝试标签推断步骤
            if batch_idx % 10 == 0 or total_samples >= 50:  # 降低阈值
                # 尝试更新类别统计信息
                if label_inference_module.update_class_stats():
                        # 更新触发器状态
                        if adversary_model.villain_trigger:
                            adversary_model.villain_trigger.update_inference_stats()
                            return True
        
        # 清空梯度避免累积
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
    
    # 最终尝试更新类别统计信息
    print(f"最终尝试标签推断初始化，已收集 {label_inference_module.get_total_samples()} 个样本")
    if label_inference_module.update_class_stats(force=True):
            # 更新触发器状态
            if adversary_model.villain_trigger:
                adversary_model.villain_trigger.update_inference_stats()
                return True
    
    return False

def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置CUDA设备和优化
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU: {args.gpu}")
        torch.cuda.set_device(args.gpu)
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        # 设置CUDA优化
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 验证GPU设备
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA不可用，使用CPU")
        global DEVICE
        DEVICE = torch.device("cpu")
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"VILLAIN 攻击训练 (带标签推断) - 数据集: {args.dataset}")
    print(f"设备: {DEVICE}")
    print(f"参与方数量: {args.party_num}")
    print(f"恶意方ID: {args.bkd_adversary}")
    print(f"目标类别: {args.target_class}")
    print(f"触发器大小: {args.trigger_size}")
    print(f"触发器强度: {args.trigger_magnitude}")
    print(f"毒化预算: {args.poison_budget}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    if args.early_stopping:
        print(f"早停: 启用 (耐心轮数={args.patience}, 监控指标={args.monitor})")
    print("="*80 + "\n")

    # 加载数据集
    train_loader, test_loader, input_shape = load_dataset(args.dataset, args.data_dir, args.batch_size, args.image_size)

    # 创建模型
    bottom_models, modelC = create_models()
    
    # 打印模型结构
    print("\n模型结构:")
    print(f"底部模型数量: {args.party_num}")
    print(f"恶意模型ID: {args.bkd_adversary}")
    for i, model in enumerate(bottom_models):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"底部模型 {i}: {total_params:,} 参数" + (" (恶意)" if i == args.bkd_adversary else ""))
    
    total_params = sum(p.numel() for p in modelC.parameters())
    print(f"顶部模型: {total_params:,} 参数")

    # 创建优化器
    if args.use_adam:
        optimizers = [optim.Adam(model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.weight_decay) 
                     for model in bottom_models]
        
        optimizerC = optim.Adam(modelC.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.weight_decay)
        print("使用Adam优化器")
    else:
        optimizers = [optim.SGD(model.parameters(), 
                               lr=args.lr, 
                               momentum=args.momentum, 
                               weight_decay=args.weight_decay) 
                     for model in bottom_models]
        
        optimizerC = optim.SGD(modelC.parameters(), 
                              lr=args.lr, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)
        print("使用SGD优化器")
    
    # 为模型添加优化器引用（用于梯度收集）
    for i, model in enumerate(bottom_models):
        model.optimizer = optimizers[i]
    
    # 改进的学习率调度器
    if args.lr_schedule == 'cosine':
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01) 
                     for optimizer in optimizers]
        schedulerC = optim.lr_scheduler.CosineAnnealingLR(optimizerC, T_max=args.epochs, eta_min=args.lr*0.01)
        print("使用余弦退火学习率调度")
    elif args.lr_schedule == 'step':
        schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
                     for optimizer in optimizers]
        schedulerC = optim.lr_scheduler.StepLR(optimizerC, step_size=30, gamma=0.5)
        print("使用阶梯学习率调度")
    else:
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='max', 
                                                         factor=0.5, 
                                                         patience=5,
                                                         verbose=True) 
                     for optimizer in optimizers]
        
        schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optimizerC, 
                                                        mode='max', 
                                                        factor=0.5, 
                                                        patience=5,
                                                        verbose=True)
        print("使用自适应学习率调度")

    # 预训练阶段: 收集标签推断数据并强制初始化推断模块
    print("\n" + "="*60)
    print("预训练阶段: 收集标签推断数据")
    print("="*60)
    
    # 获取恶意模型
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 确保恶意模型有触发器
    if not hasattr(adversary_model, 'villain_trigger') or adversary_model.villain_trigger is None:
        print("\n警告: 恶意模型触发器未正确设置，尝试重新设置...")
        villain_trigger = VILLAINTrigger(args)
        adversary_model.set_villain_trigger(villain_trigger)
    
    # 确保推断模块得到初始化
    success = collect_inference_data(modelC, bottom_models, train_loader, args)
    
    # 显示训练配置
    print("\n" + "="*40)
    print("开始训练")
    print(f"轮次: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"优化器: SGD (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
    print(f"后门启动轮次: {args.Ebkd}")
    print(f"标签推断状态: {'已初始化' if adversary_model.label_inference.initialized else '未初始化'}")
    print(f"触发器状态: {'已初始化' if (hasattr(adversary_model, 'villain_trigger') and adversary_model.villain_trigger.is_initialized) else '未初始化'}")
    print("="*40 + "\n")

    # 训练循环
    best_accuracy = 0
    best_inference_acc = 0
    best_asr = 0
    best_epoch = 0
    
    # 存储最佳模型对应的所有指标
    best_metrics = {
        'test_acc': 0,
        'inference_acc': 0,
        'asr': 0,
        'epoch': 0
    }
    
    no_improvement_count = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        # 训练一个epoch
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args
        )

        # 测试
        test_loss, test_acc, _, test_inference_acc = test(
            modelC, bottom_models, test_loader, is_backdoor=False, epoch=epoch, args=args
        )

        # 后门测试
        bkd_loss, bkd_acc, true_asr, bkd_inference_acc = test(
            modelC, bottom_models, test_loader, is_backdoor=True, epoch=epoch, args=args
        )
        
        # 更新学习率
        if args.lr_schedule in ['cosine', 'step']:
            # 对于cosine和step调度器，每个epoch更新
            for scheduler in schedulers:
                scheduler.step()
            schedulerC.step()
        else:
            # 对于plateau调度器，根据性能更新
            for scheduler in schedulers:
                scheduler.step(test_acc)
            schedulerC.step(test_acc)

        # 打印训练信息
        print(f"\nEpoch {epoch} 结果:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train ASR: {train_asr:.2f}%, Train Inference Acc: {train_inference_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Inference Acc: {test_inference_acc:.2f}%")
        print(f"Backdoor Loss: {bkd_loss:.4f}, Backdoor Acc: {bkd_acc:.2f}%, True ASR: {true_asr:.2f}%, Backdoor Inference Acc: {bkd_inference_acc:.2f}%")

        # 强制进行标签推断更新 (每10个epoch)
        if epoch % 10 == 0 and adversary_model.label_inference is not None:
            print("\n强制更新标签推断...")
            # 强制更新类别统计信息
            if hasattr(adversary_model.label_inference, 'update_class_stats'):
                adversary_model.label_inference.update_class_stats(force=True)
            
            # 打印当前状态
            if adversary_model.label_inference.initialized:
                print(f"标签推断状态: 已初始化")
                if hasattr(adversary_model.label_inference, 'get_total_samples'):
                    print(f"梯度历史大小: {adversary_model.label_inference.get_total_samples()}")
                else:
                    print(f"梯度历史大小: {len(adversary_model.label_inference.gradient_history)}")
                
                # 更新触发器状态
                if hasattr(adversary_model, 'villain_trigger'):
                    adversary_model.villain_trigger.update_inference_stats()
        
        # 组合分数标准
        combined_score = 0.5 * test_acc + 0.5 * true_asr
        best_combined_score = 0.5 * best_metrics['test_acc'] + 0.5 * best_metrics['asr']
        print(f"组合分数: 当前={combined_score:.2f}, 最佳={best_combined_score:.2f}")
        
        # 检查是否需要保存新的最佳模型
        if combined_score > best_combined_score:
            # 更新所有最佳指标
            best_metrics = {
                'test_acc': test_acc,
                'inference_acc': test_inference_acc,
                'asr': true_asr,
                'epoch': epoch
            }
            best_accuracy = test_acc
            best_inference_acc = test_inference_acc
            best_asr = true_asr
            best_epoch = epoch
            # 保存最佳模型
            save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr, test_inference_acc)
            print(f"\n保存最佳模型 (Epoch {epoch}) (组合分数提升)")
            print(f"Clean Acc: {test_acc:.2f}%, Inference Acc: {test_inference_acc:.2f}%, ASR: {true_asr:.2f}%")
            # 重置早停计数器
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"\n没有改进: {no_improvement_count}/{args.patience} (最佳组合分数: {best_combined_score:.2f} 在 Epoch {best_epoch})")
            # 检查是否达到早停条件
            if args.early_stopping and no_improvement_count >= args.patience:
                print(f"\n早停触发! {args.patience} 轮次内没有改进。")
                print(f"最佳模型 (Epoch {best_epoch}):")
                print(f"Clean Acc: {best_accuracy:.2f}%, Inference Acc: {best_inference_acc:.2f}%, ASR: {best_asr:.2f}%")
                break
    
    # 训练结束，输出详细的最佳结果
    print("\n" + "="*60)
    print(f"训练完成！最佳模型 (Epoch {best_metrics['epoch']}):")
    print(f"Clean Accuracy: {best_metrics['test_acc']:.2f}%")
    print(f"Inference Accuracy: {best_metrics['inference_acc']:.2f}%")
    print(f"Attack Success Rate: {best_metrics['asr']:.2f}%")
    print("="*60)
    
    print("\n使用NUS-WIDE数据集的VILLAIN攻击运行命令示例:")
    print(f"python train_nuswide_villain_with_inference.py --dataset NUSWIDE --data-dir './data/NUS-WIDE' --batch-size 32 --epochs 100 --early-stopping --patience 15 --monitor inference_acc --gpu 0 --image-size 224 --num-classes 21")
    print("\n[VFL优化] 使用NUS-WIDE数据集的VILLAIN攻击运行命令示例:")
    print("1. VFL垂直联邦学习模式 (5个概念，单标签分类):")
    print(f"python train_nuswide_villain_with_inference.py --dataset NUSWIDE --data-dir './data/NUS-WIDE' --batch-size 64 --epochs 50 --early-stopping --patience 10 --monitor test_acc --gpu 0 --image-size 224 --selected-concepts buildings grass animal water person")
    print("\n2. VFL多标签模式 (5个概念，多标签分类):")
    print(f"python train_nuswide_villain_with_inference.py --dataset NUSWIDE --data-dir './data/NUS-WIDE' --batch-size 64 --epochs 50 --early-stopping --patience 10 --monitor test_acc --gpu 0 --image-size 224 --multi-label --selected-concepts buildings grass animal water person")
    print("\n3. VFL优化训练模式 (快速训练):") 
    print(f"python train_nuswide_villain_with_inference.py --dataset NUSWIDE --selected-concepts buildings grass animal water person --batch-size 128 --epochs 30 --lr 0.02 --early-stopping --patience 8")
    print("\n[VFL说明] 默认配置已优化用于垂直联邦学习场景，包括:")
    print("- 默认使用5个概念：buildings, grass, animal, water, person")
    print("- 单标签分类模式 (更适合VFL场景)")
    print("- 简化的网络结构和数据增强")
    print("- 优化的批次大小和学习率")
    print("- 降低的数据量以提高训练效率")

if __name__ == '__main__':
    main() 