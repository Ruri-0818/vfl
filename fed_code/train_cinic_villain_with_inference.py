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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
import time
from PIL import Image
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("villain_training.log")
    ]
)
logger = logging.getLogger("VILLAIN")

# 设置命令行参数
parser = argparse.ArgumentParser(description='针对CINIC-10数据集的VILLAIN攻击训练 (带标签推断)')
# 原有参数
parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=float, default=0.75, help='VILLAIN触发器大小(占恶意方嵌入向量维度的比例)')
parser.add_argument('--trigger-magnitude', type=float, default=3.0, help='VILLAIN触发器强度')
parser.add_argument('--position', type=str, default='mid', help='恶意参与方位置')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
parser.add_argument('--min-epochs', type=int, default=50, help='最小训练轮数')
parser.add_argument('--max-epochs', type=int, default=300, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=5.0, help='后门损失权重')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.5, help='毒化预算')
parser.add_argument('--Ebkd', type=int, default=1, help='后门注入开始轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.5, help='学习率倍增器')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量 (10)')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='./data', help='数据集目录')

# 标签推断相关参数
parser.add_argument('--inference-weight', type=float, default=0.1, help='标签推断损失权重')
parser.add_argument('--history-size', type=int, default=1000, help='嵌入向量历史记录大小')
parser.add_argument('--cluster-update-freq', type=int, default=50, help='聚类更新频率(批次)')
parser.add_argument('--inference-start-epoch', type=int, default=10, help='开始标签推断的轮数')
parser.add_argument('--confidence-threshold', type=float, default=0.8, help='标签推断置信度阈值')
parser.add_argument('--adaptive-threshold', action='store_true', help='是否使用自适应置信度阈值')
parser.add_argument('--feature-selection', action='store_true', help='是否启用特征选择以提高推断准确率')
parser.add_argument('--use-ensemble', action='store_true', help='是否使用集成方法提高推断准确率')

# 早停参数
parser.add_argument('--early-stopping', action='store_true', help='是否使用早停')
parser.add_argument('--monitor', type=str, default='inference_acc', choices=['test_acc', 'inference_acc', 'asr'], 
                    help='用于早停的监控指标')

# 新增ASR监控与优化参数
parser.add_argument('--asr-target', type=float, default=95.0, help='目标ASR值，达到此值提前停止训练')
parser.add_argument('--balance-metrics', action='store_true', help='平衡模型准确率与ASR，防止过拟合')
parser.add_argument('--adaptive-budget', action='store_true', help='自适应调整毒化预算')
parser.add_argument('--advanced-trigger', action='store_true', help='使用高级触发器生成策略')

# 在命令行参数部分添加新参数
parser.add_argument('--save-optimizer-state', action='store_true', help='是否保存优化器状态以支持恢复训练')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 定义CINIC10数据集类
class CINIC10Dataset(Dataset):
    """CINIC10数据集加载器"""
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, 'CINIC10')
        self.split = split  # 'train', 'valid', 或 'test'
        self.transform = transform
        
        # CINIC10类别
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                         'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 读取数据
        self.img_paths = []
        self.targets = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root, split, class_name)
            if not os.path.exists(class_dir):
                raise RuntimeError(f'Class directory {class_dir} not found. Make sure the CINIC10 dataset is correctly downloaded and structured.')
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.targets.append(class_idx)
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f'No images found in {split} split. Check your dataset path.')
        
        logger.info(f"Loaded {len(self.img_paths)} images from {split} split.")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        target = self.targets[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, target 

# 标签推断器实现
class LabelInferenceModule:
    """增强版标签推断模块，使用多种分类器和特征选择技术提高推断准确率"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.history_size = args.history_size
        self.confidence_threshold = args.confidence_threshold
        self.adaptive_threshold = args.adaptive_threshold
        self.cluster_update_freq = args.cluster_update_freq
        self.use_ensemble = args.use_ensemble
        self.feature_selection = args.feature_selection
        
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
        
        # 自适应置信度阈值
        self.min_threshold = 0.5
        self.max_threshold = 0.95
        self.current_threshold = self.confidence_threshold
        
        logger.info(f"增强版标签推断模块创建完成: 特征维度={feature_dim}, 类别数={num_classes}, "
                  f"历史大小={self.history_size}, 初始置信度阈值={self.confidence_threshold}")
        
        if self.adaptive_threshold:
            logger.info(f"自适应置信度阈值: 已启用 (范围={self.min_threshold}-{self.max_threshold})")
        
        if self.use_ensemble:
            logger.info(f"集成推断: 已启用 (权重={self.ensemble_weights})")
            
        if self.feature_selection:
            logger.info(f"特征选择: 已启用")
    
    def get_total_samples(self):
        """获取总样本数"""
        return sum(len(samples) for samples in self.data_by_class.values())
    
    def get_samples_per_class(self):
        """获取每个类别的样本数"""
        return {c: len(samples) for c, samples in self.data_by_class.items()}
    
    def update_history(self, features, labels):
        """更新历史数据，保持每个类别的样本平衡"""
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 增加的样本数计数
        added_samples = 0
        
        # 计算每个类别的当前样本数和目标样本数
        class_counts = self.get_samples_per_class()
        max_samples_per_class = self.history_size // self.num_classes
        
        # 平衡系数: 对样本数少的类别更有可能接受新样本
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
            else:
                # 已达到上限，随机替换旧样本
                # 样本多样性策略：优先保留离质心较远的样本
                if np.random.random() < 0.3:  # 30%的替换概率
                    if self.means is not None and label_int in self.means:
                        # 计算与均值的距离
                        mean_vector = self.means[label_int]
                        distances = [np.linalg.norm(x - mean_vector) for x in self.data_by_class[label_int]]
                        
                        # 找出最近的样本索引 (最不具代表性的样本)
                        closest_idx = np.argmin(distances)
                        self.data_by_class[label_int][closest_idx] = feature
                    else:
                        # 如果没有均值信息，则随机替换
                        idx = np.random.randint(0, len(self.data_by_class[label_int]))
                        self.data_by_class[label_int][idx] = feature
                    
                    added_samples += 1
        
        return added_samples
    
    def _select_features(self, X_train, y_train):
        """使用特征重要性进行特征选择"""
        if not self.feature_selection or X_train.shape[1] <= 10:
            return np.arange(X_train.shape[1]), X_train
        
        try:
            from sklearn.feature_selection import SelectFromModel
            # 使用L1正则化的逻辑回归作为特征选择器
            selector = SelectFromModel(
                LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=200, class_weight='balanced'),
                max_features=min(X_train.shape[1] // 2, 50)  # 最多选择一半特征，但不超过50个
            )
            selector.fit(X_train, y_train)
            
            # 获取选择的特征索引
            selected_indices = np.where(selector.get_support())[0]
            
            # 如果选择的特征太少，退回到所有特征
            if len(selected_indices) < 5:
                logger.warning(f"特征选择返回太少特征 ({len(selected_indices)}), 使用所有特征")
                return np.arange(X_train.shape[1]), X_train
            
            # 应用特征选择
            X_train_selected = X_train[:, selected_indices]
            logger.info(f"特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]} 特征")
            
            return selected_indices, X_train_selected
        except Exception as e:
            logger.warning(f"特征选择失败: {str(e)}, 使用所有特征")
            return np.arange(X_train.shape[1]), X_train
    
    def _adjust_confidence_threshold(self):
        """根据历史准确率自适应调整置信度阈值"""
        if not self.adaptive_threshold or len(self.accuracy_history) < 3:
            return
        
        recent_accuracy = np.mean(self.accuracy_history[-3:])
        
        # 如果准确率高，可以降低阈值接受更多预测
        if recent_accuracy > 85:
            self.current_threshold = max(self.min_threshold, self.current_threshold - 0.05)
        # 如果准确率低，提高阈值只接受高置信度预测
        elif recent_accuracy < 70:
            self.current_threshold = min(self.max_threshold, self.current_threshold + 0.05)
        
        logger.info(f"自适应置信度阈值调整: {self.current_threshold:.2f} (基于近期准确率 {recent_accuracy:.1f}%)")
    
    def update_class_stats(self, force=False):
        """更新类别统计信息和分类器"""
        self.update_counter += 1
        
        # 如果没有min_samples属性，设置一个默认值
        if not hasattr(self, 'min_samples'):
            self.min_samples = max(100, 10 * self.num_classes) 
            logger.info(f"设置默认最小样本数: {self.min_samples}")
            
        # 检查是否每个类别都有足够的样本
        class_counts = self.get_samples_per_class()
        min_samples = min(class_counts.values()) if class_counts else 0
        
        # 如果没有强制更新，且样本数不足，则跳过
        if not force and min_samples < self.min_samples_per_class:
            logger.info(f"更新类别统计失败: 最小样本数 {min_samples} < {self.min_samples_per_class}")
            return False
        
        # 如果强制更新且有些类别样本不足，调整要求
        if force and min_samples < self.min_samples_per_class:
            adjusted_min = max(5, min_samples)
            logger.info(f"强制更新: 调整最小样本数要求 {self.min_samples_per_class} -> {adjusted_min}")
            self.min_samples_per_class = adjusted_min
        
        # 确保每个类别都有样本
        valid_classes = [c for c in range(self.num_classes) if len(self.data_by_class.get(c, [])) >= self.min_samples_per_class]
        
        if len(valid_classes) < 2:  # 至少需要两个类别才能训练分类器
            logger.warning(f"更新类别统计失败: 只有 {len(valid_classes)}/{self.num_classes} 个类别有足够样本")
            return False
        
        # 准备训练数据
        X_train = []
        y_train = []
        
        for c in valid_classes:
            # 当前类别的样本
            class_samples = np.array(self.data_by_class[c])
            class_labels = np.array(self.labels_by_class[c])
            
            # 添加噪声增强鲁棒性
            if len(class_samples) > 10:
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, class_samples.shape)
                class_samples = class_samples + noise
            
            # 计算均值 - 用于特征重要性评估和样本多样性
            mean = np.mean(class_samples, axis=0)
            if self.means is None:
                self.means = {}
            self.means[c] = mean
            
            # 将数据添加到训练集
            X_train.append(class_samples)
            y_train.append(class_labels)
        
        # 合并所有类别的数据
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        # 应用特征选择 (如果启用)
        if self.feature_selection:
            self.selected_features, X_train_selected = self._select_features(X_train, y_train)
        else:
            X_train_selected = X_train
            self.selected_features = np.arange(X_train.shape[1])
        
        # 训练分类器
        try:
            from sklearn.linear_model import LogisticRegression
            
            # 保存旧分类器作为备份
            if self.classifier is not None and self.use_ensemble:
                self.backup_classifier = self.classifier
            
            # 训练新的主分类器
            self.classifier = LogisticRegression(
                C=1.0, max_iter=1000, tol=1e-4, solver='lbfgs', multi_class='multinomial',
                class_weight='balanced'
            )
            self.classifier.fit(X_train_selected, y_train)
            
            # 计算训练集准确率
            train_accuracy = self.classifier.score(X_train_selected, y_train)
            logger.info(f"分类器训练完成，训练集准确率: {train_accuracy*100:.2f}%")
            
            # 记录准确率历史
            self.accuracy_history.append(train_accuracy * 100)
            
            # 调整置信度阈值 (如果启用自适应)
            self._adjust_confidence_threshold()
            
            # 计算特征重要性
            self.feature_importance = np.abs(self.classifier.coef_).mean(axis=0)
            
            self.initialized = True
            logger.info(f"类别统计信息更新成功: {len(valid_classes)}/{self.num_classes} 类别有效")
            return True
        except Exception as e:
            logger.error(f"分类器训练失败: {str(e)}")
            self.classifier = None
            self.initialized = False
            return False
    
    def infer_labels(self, features):
        """使用训练好的分类器推断标签"""
        if not self.initialized or self.classifier is None:
            return None, None
        
        features_np = features.detach().cpu().numpy()
        
        # 应用特征选择 (如果已启用)
        if self.selected_features is not None and len(self.selected_features) < features_np.shape[1]:
            features_np = features_np[:, self.selected_features]
        
        # 预测类别和概率
        try:
            # 主分类器预测
            pred_probs = self.classifier.predict_proba(features_np)
            
            # 如果启用集成且备份分类器可用
            if self.use_ensemble and self.backup_classifier is not None:
                try:
                    # 获取备份分类器预测
                    backup_probs = self.backup_classifier.predict_proba(features_np)
                    
                    # 加权平均两个分类器的预测
                    pred_probs = self.ensemble_weights[0] * pred_probs + self.ensemble_weights[1] * backup_probs
                except Exception as e:
                    logger.warning(f"备份分类器预测失败: {str(e)}")
            
            pred_labels = np.argmax(pred_probs, axis=1)
            confidence = np.max(pred_probs, axis=1)
            
            # 使用当前阈值 (可能是自适应的)
            threshold = self.current_threshold if self.adaptive_threshold else self.confidence_threshold
            
            # 对置信度低的预测应用阈值
            for i in range(len(pred_labels)):
                if confidence[i] < threshold:
                    pred_labels[i] = -1  # 置信度不足，标记为未知
            
            # 更新置信度历史
            if len(confidence) > 0:
                self.confidence_history.append(np.mean(confidence))
            
            return pred_labels.tolist(), confidence.tolist()
        except Exception as e:
            logger.error(f"标签推断失败: {str(e)}")
            return None, None

    def get_metrics(self):
        """返回当前推断模块的性能指标"""
        metrics = {
            "initialized": self.initialized,
            "total_samples": self.get_total_samples(),
            "samples_per_class": self.get_samples_per_class(),
            "average_confidence": np.mean(self.confidence_history[-10:]) if len(self.confidence_history) >= 10 else 0,
            "average_accuracy": np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else 0,
            "current_threshold": self.current_threshold if self.adaptive_threshold else self.confidence_threshold
        }
        return metrics

# 扩展VILLAIN触发器实现
class VILLAINTrigger:
    """VILLAIN攻击使用的触发器实现，与标签推断模块配合工作"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.trigger_magnitude = args.trigger_magnitude
        self.trigger_size = args.trigger_size
        self.original_magnitude = args.trigger_magnitude
        self.original_size = args.trigger_size
        self.feature_indices = None
        self.adversary_id = args.bkd_adversary
        self.label_inference = None
        self.is_initialized = False
        self.batch_count = 0
        
        # 自适应预算和平衡指标
        self.use_adaptive_budget = hasattr(args, 'adaptive_budget') and args.adaptive_budget
        self.use_balance_metrics = hasattr(args, 'balance_metrics') and args.balance_metrics
        self.use_advanced_trigger = hasattr(args, 'advanced_trigger') and args.advanced_trigger
        
        # 跟踪统计数据
        self.last_inf_acc = 0
        self.last_asr = 0
        self.feature_importance = None
        
        logger.info(f"创建VILLAIN触发器 (大小={self.trigger_size}, 强度={self.trigger_magnitude})")
        if self.use_adaptive_budget:
            logger.info("启用自适应预算")
        if self.use_balance_metrics:
            logger.info("启用平衡指标")
        if self.use_advanced_trigger:
            logger.info("启用高级触发器策略")
    
    def set_label_inference(self, label_inference):
        """设置标签推断模块引用"""
        self.label_inference = label_inference
        # 如果标签推断模块已初始化，同步状态
        if label_inference and hasattr(label_inference, 'initialized') and label_inference.initialized:
            self.is_initialized = True
            logger.info("触发器已连接到初始化的标签推断模块")
    
    def update_inference_stats(self, inference_acc=None, asr=None):
        """更新触发器状态，并根据需要调整触发器参数"""
        # 更新初始化状态
        if self.label_inference and hasattr(self.label_inference, 'initialized') and self.label_inference.initialized:
            self.is_initialized = True
            
            # 如果启用了自适应参数
            if inference_acc is not None:
                self.last_inf_acc = inference_acc
            
            if asr is not None:
                self.last_asr = asr
                
                # 根据ASR和推断准确率调整触发器参数
                if self.use_adaptive_budget and hasattr(self.args, 'poison_budget'):
                    self._adjust_poison_budget(asr)
                
                if self.use_balance_metrics and self.last_inf_acc > 0:
                    self._adjust_trigger_params(asr, inference_acc)
                    
                if self.use_advanced_trigger and self.batch_count % 50 == 0:
                    self._update_feature_importance()
    
    def _adjust_poison_budget(self, asr):
        """根据当前ASR动态调整毒化预算"""
        if asr > 99.0:  # ASR非常高，可以降低预算
            new_budget = max(0.05, self.args.poison_budget * 0.8)
            if new_budget != self.args.poison_budget:
                logger.info(f"降低毒化预算: {self.args.poison_budget:.2f} -> {new_budget:.2f} (ASR={asr:.1f}%)")
                self.args.poison_budget = new_budget
        elif asr < 80.0:  # ASR不够高，增加预算
            new_budget = min(0.5, self.args.poison_budget * 1.2)
            if new_budget != self.args.poison_budget:
                logger.info(f"增加毒化预算: {self.args.poison_budget:.2f} -> {new_budget:.2f} (ASR={asr:.1f}%)")
                self.args.poison_budget = new_budget
    
    def _adjust_trigger_params(self, asr, inference_acc):
        """平衡ASR和推断准确率，调整触发器参数"""
        # 如果ASR高但推断准确率低，减小触发器强度
        if asr > 95.0 and inference_acc < 30.0:
            self.trigger_magnitude = max(0.5, self.trigger_magnitude * 0.9)
            logger.info(f"减小触发器强度: {self.trigger_magnitude:.2f} (平衡ASR和推断)")
        # 如果ASR低，增加触发器强度
        elif asr < 85.0:
            self.trigger_magnitude = min(self.original_magnitude * 1.5, self.trigger_magnitude * 1.1)
            logger.info(f"增加触发器强度: {self.trigger_magnitude:.2f} (提高ASR)")
    
    def _update_feature_importance(self):
        """基于推断模块的数据更新特征重要性"""
        if not self.label_inference or not self.label_inference.initialized:
            return
        
        # 获取标签推断模块中的特征信息
        if hasattr(self.label_inference, 'means') and self.label_inference.means:
            # 计算特征重要性
            feature_dim = self.label_inference.feature_dim
            importance = np.zeros(feature_dim)
            
            # 使用类别均值信息计算每个特征的区分度
            if self.target_class in self.label_inference.means:
                target_mean = self.label_inference.means[self.target_class]
                
                for c in range(self.args.num_classes):
                    if c != self.target_class and c in self.label_inference.means:
                        other_mean = self.label_inference.means[c]
                        # 计算差异
                        diff = np.abs(target_mean - other_mean)
                        importance += diff
                
                # 选择最重要的特征
                if np.sum(importance) > 0:
                    num_features = int(feature_dim * self.trigger_size)
                    top_indices = np.argsort(importance)[-num_features:]
                    self.feature_indices = torch.tensor(top_indices, device=DEVICE)
                    logger.info(f"基于特征重要性创建新触发器模式 (共{len(top_indices)}个特征)")
                    self.feature_importance = importance
    
    def _select_target_features(self, embeddings):
        """选择要注入触发器的特征"""
        embed_dim = embeddings.size(1)
        device = embeddings.device
        
        # 确定要修改的特征数量
        num_features = int(embed_dim * self.trigger_size)
        
        # 如果没有预先计算的特征索引或需要更新
        if self.feature_indices is None or len(self.feature_indices) != num_features:
            if self.use_advanced_trigger and self.feature_importance is not None:
                # 使用特征重要性选择
                top_indices = np.argsort(self.feature_importance)[-num_features:]
                self.feature_indices = torch.tensor(top_indices, device=device)
            else:
                # 默认使用前num_features个特征
                self.feature_indices = torch.arange(num_features, device=device)
        
        return self.feature_indices

    def construct_trigger(self, embeddings, inferred_labels=None):
        """构建触发器，可选地基于标签推断结果"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 创建触发器掩码
        trigger_mask = torch.zeros_like(embeddings)
        
        # 获取要修改的特征索引
        feature_indices = self._select_target_features(embeddings)
        
        # 判断是否有有效的标签推断结果
        if inferred_labels is not None:
            # 对每个样本个性化应用触发器
            for i in range(batch_size):
                if i < len(inferred_labels) and inferred_labels[i] != self.target_class:
                    # 非目标类样本应用触发器
                    trigger_mask[i, feature_indices] = self.trigger_magnitude
        else:
            # 没有推断标签，对所有样本应用相同的触发器
            trigger_mask[:, feature_indices] = self.trigger_magnitude
            
        self.batch_count += 1
        return trigger_mask

# 添加ResBlock和SELayer的定义
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
        
        # 添加SE层
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

class CINICBottomModel(nn.Module):
    """CINIC底部模型，增强版支持高级标签推断和触发器注入"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(CINICBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 使用ResNet风格的特征提取器
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # 计算特征图大小
        feature_dim = 128 * (32 // 4) * (32 // 4)  # 经过2次stride=2的下采样
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
        # 如果是恶意模型，初始化标签推断模块
        if is_adversary and args is not None:
            self.label_inference = None  # 稍后初始化
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # 第一个块可能需要下采样
        layers.append(ResBlock(in_channels, out_channels, stride))
        # 其余块不需要下采样
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def set_villain_trigger(self, villain_trigger):
        """设置VILLAIN触发器"""
        if self.is_adversary:
            self.villain_trigger = villain_trigger
            # 创建标签推断模块
            if not hasattr(self, 'label_inference') or self.label_inference is None:
                logger.info("为恶意模型创建标签推断模块")
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
                logger.info("标签推断模块已初始化，正在更新触发器状态...")
                self.villain_trigger.update_inference_stats()
    
    def forward(self, x, attack_flags=None, inferred_labels=None):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (8, 8))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        features = self.classifier(x)
        
        # 攻击逻辑
        if self.is_adversary and attack_flags is not None and torch.any(attack_flags):
            # 仅对攻击样本应用触发器
            attack_mask = attack_flags
            if hasattr(self, 'villain_trigger'):
                # 构建触发器 (使用推断的标签)
                trigger = self.villain_trigger.construct_trigger(features, inferred_labels)
                # 只修改攻击样本
                features[attack_mask] = features[attack_mask] + trigger[attack_mask]
        
        return features

class CINICTopModel(nn.Module):
    """CINIC-10顶部模型"""
    def __init__(self, input_dim=256, num_classes=10):
        super(CINICTopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
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

def train_epoch(modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args):
    """训练一个轮次，包括标准VILLAIN标签推断和后门注入"""
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if hasattr(adversary_model, 'label_inference') else None
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    attack_success = 0
    attack_samples = 0
    
    # 推断准确率统计
    inference_accuracies = []
    total_inference_correct = 0
    total_inference_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    target_class = args.target_class
    
    # 确定是否进行梯度收集 - 每2个epoch进行一次
    collect_gradients = label_inference_module is not None and (epoch % 2 == 0)
    
    # 记录批次数
    batch_count = 0
    warmup_batches = min(50, len(train_loader) // 10)  # 热身批次数
    
    for batch_idx, (data, target) in enumerate(pbar):
        # 增加批次计数
        batch_count += 1
        data, target = data.to(DEVICE), target.to(DEVICE)
        batch_size = data.size(0)
        total_samples += batch_size
        
        # 清除梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # 准备后门数据
        is_backdoor_batch = (epoch >= args.Ebkd)
        # 决定是否应用后门攻击
        if is_backdoor_batch:
            bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target, args)
            backdoor_batch_samples = attack_flags.sum().item()
            attack_samples += backdoor_batch_samples
        else:
            bkd_data, bkd_target, attack_flags = data, target, torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
            backdoor_batch_samples = 0
        
        # 前向传播 - 干净数据和后门数据
        bottom_outputs = []
        for i, model in enumerate(bottom_models):
            if i == args.bkd_adversary and is_backdoor_batch:
                if hasattr(model, 'label_inference') and model.label_inference and model.label_inference.initialized:
                    # 获取特征用于推断
                    original_data = data.view(data.size(0), -1)
                    inferred_labels, _ = model.label_inference.infer_labels(original_data)
                    
                    # 应用触发器，同时传递推断的标签
                    output = model(bkd_data, attack_flags=attack_flags, inferred_labels=inferred_labels)
                else:
                    # 没有推断标签时使用普通触发器
                    output = model(bkd_data, attack_flags=attack_flags)
            else:
                # 其他模型正常处理
                output = model(data)
            
            bottom_outputs.append(output)
        
        # 拼接底部模型输出
        combined_output = torch.cat(bottom_outputs, dim=1)
        
        # 顶部模型前向传播
        output = modelC(combined_output)
        
        # 根据是否攻击计算损失
        if is_backdoor_batch and backdoor_batch_samples > 0:
            # 修改目标样本标签为目标类
            modified_target = target.clone()
            modified_target[attack_flags] = target_class
            loss = criterion(output, modified_target)
        else:
            # 使用原始标签计算损失
            loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 恶意方收集梯度信息用于标签推断
        if collect_gradients and label_inference_module and batch_count <= warmup_batches:
            if hasattr(adversary_model, 'get_saved_data'):
                saved_data, saved_grad = adversary_model.get_saved_data()
                if saved_data is not None and saved_grad is not None:
                    # 更新标签推断历史
                    original_data = saved_data.view(saved_data.size(0), -1)
                    samples_added = label_inference_module.update_history(original_data, saved_grad)
                    
                    if batch_count % 10 == 0:  # 更频繁地输出信息
                        logger.info(f"梯度收集: 批次 {batch_count}, 累计样本: {label_inference_module.get_total_samples()}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(modelC.parameters(), 1.0)
        for model in bottom_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 优化
        for optimizer in optimizers:
            optimizer.step()
        optimizerC.step()
        
        # 统计分类准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total_correct += correct
        total_loss += loss.item() * batch_size
        
        # 评估标签推断准确率
        if epoch >= args.inference_start_epoch and hasattr(adversary_model, 'label_inference') and adversary_model.label_inference.initialized:
            # 仅对干净样本进行推断
            clean_mask = ~attack_flags
            clean_count = clean_mask.sum().item()
            
            if clean_count > 0 and batch_idx % 3 == 0:  # 每3个批次评估一次
                with torch.no_grad():  # 确保不计算额外的梯度
                    clean_outputs = bottom_outputs[args.bkd_adversary][clean_mask]
                    clean_targets = target[clean_mask]
                    
                    # 使用标签推断预测
                    original_data = data[clean_mask].view(clean_count, -1)
                    inferred_labels, _ = adversary_model.label_inference.infer_labels(original_data)
                    
                    if inferred_labels is not None:
                        # 计算推断准确率 - 关注推断对目标类的检测能力
                        correct_preds = 0
                        for j, (pred, true) in enumerate(zip(inferred_labels, clean_targets.cpu().numpy())):
                            is_target_class = (true == target_class)
                            
                            # 判断预测是否正确：
                            # - 真实是目标类且预测为目标类(pred == target_class)
                            # - 真实不是目标类且预测不是目标类(pred != target_class)
                            if (is_target_class and pred == target_class) or (not is_target_class and pred != target_class):
                                correct_preds += 1
                        
                        # 更新推断准确率统计
                        inf_acc = 100.0 * correct_preds / clean_count
                        inference_accuracies.append(inf_acc)
                        total_inference_correct += correct_preds
                        total_inference_samples += clean_count
        
        # 计算攻击成功率 - 仅对攻击样本
        if is_backdoor_batch and backdoor_batch_samples > 0:
            # 确保pred是张量而不是标量
            if isinstance(pred, int) or (isinstance(pred, torch.Tensor) and pred.dim() == 0):
                # 如果pred是标量，将其转为正确的形状
                attack_pred = torch.full_like(attack_flags, pred, dtype=torch.long)
            else:
                # 否则正常索引
                attack_pred = pred[attack_flags]
            
            attack_target = torch.ones_like(attack_pred) * target_class
            batch_attack_success = attack_pred.eq(attack_target).sum().item()
            attack_success += batch_attack_success
            
            # 计算攻击成功率 - 检查除零错误
            if backdoor_batch_samples > 0:
                batch_asr = 100.0 * batch_attack_success / backdoor_batch_samples
            else:
                batch_asr = 0.0
        
        # 更新进度条 - 检查除零错误
        if is_backdoor_batch:
            current_asr = 100. * attack_success / attack_samples if attack_samples > 0 else 0.0
            current_inf_acc = 0.0
            if total_inference_samples > 0:
                current_inf_acc = 100. * total_inference_correct / total_inference_samples
            
            pbar.set_postfix({
                'ASR': f"{current_asr:.2f}%", 
                'Inf_Acc': f"{current_inf_acc:.2f}%"
            })
        else:
            # 更新干净测试进度条 - 检查除零错误
            current_acc = 100. * correct / ((pbar.n + 1) * batch_size) if ((pbar.n + 1) * batch_size) > 0 else 0.0
            current_inf_acc = 0.0
            if total_inference_samples > 0:
                current_inf_acc = 100. * total_inference_correct / total_inference_samples
            
            pbar.set_postfix({
                'Acc': f"{current_acc:.2f}%",
                'Inf_Acc': f"{current_inf_acc:.2f}%"
            })
    
        # 定期更新类别统计信息
        if batch_idx % args.cluster_update_freq == 0 and batch_idx > 0 and epoch >= args.inference_start_epoch:
            # 确保收集了足够多的样本
            if hasattr(adversary_model, 'label_inference') and adversary_model.label_inference:
                total_collected = adversary_model.label_inference.get_total_samples()
                min_required = min(100, 20 * args.num_classes)
                
                if total_collected > min_required:
                    logger.info(f"\n尝试更新标签推断统计信息 (批次 {batch_idx})...")
                    success = adversary_model.label_inference.update_class_stats()
                    if success:
                        logger.info("统计信息更新成功!")
                        
                        # 计算当前推断准确率
                        current_inf_acc = 0.0
                        if inference_accuracies:
                            current_inf_acc = sum(inference_accuracies) / len(inference_accuracies)
                        logger.info(f"当前推断准确率: {current_inf_acc:.2f}%")
                        
                        # 更新触发器
                        if hasattr(adversary_model, 'villain_trigger'):
                            # 计算ASR以便触发器能够进行自适应调整
                            current_asr = 100.0 * attack_success / max(1, attack_samples)
                            adversary_model.villain_trigger.update_inference_stats(current_inf_acc, current_asr)
                    else:
                        logger.info("统计信息更新失败，样本数可能不足")
        
        # 更新进度条描述
        current_inf_acc = 0.0
        if total_inference_samples > 0:
            current_inf_acc = 100.0 * total_inference_correct / total_inference_samples
        elif inference_accuracies:
            current_inf_acc = sum(inference_accuracies) / len(inference_accuracies)
        
        current_asr = 0.0
        if attack_samples > 0:
            current_asr = 100.0 * attack_success / attack_samples
            
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.0 * correct / batch_size:.1f}%",
            'inf_acc': f"{current_inf_acc:.1f}%",
            'ASR': f"{current_asr:.1f}%"
        })
    
    # 计算最终指标
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    # 计算推断准确率
    inference_accuracy = 0.0
    if total_inference_samples > 0:
        inference_accuracy = 100.0 * total_inference_correct / total_inference_samples
    elif inference_accuracies:
        inference_accuracy = sum(inference_accuracies) / len(inference_accuracies)
    
    # 计算ASR
    attack_success_rate = 0.0
    if attack_samples > 0:
        attack_success_rate = 100.0 * attack_success / attack_samples
    
    return avg_loss, accuracy, inference_accuracy, attack_success_rate

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None):
    """测试模型性能，包括干净准确率和后门攻击成功率"""
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    test_loss = 0
    correct = 0
    inference_correct = 0
    inference_samples = 0
    
    # 攻击相关指标
    attack_success = 0
    attack_total = 0
    class_success = {c: 0 for c in range(args.num_classes)}
    class_total = {c: 0 for c in range(args.num_classes)}
    asr_confidence = []
    
    target_class = args.target_class
    criterion = nn.CrossEntropyLoss()
    
    # 获取恶意模型
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 使用没有梯度的环境
    pbar = tqdm(test_loader, desc="测试" + (" (后门)" if is_backdoor else " (干净)"))
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)
            original_target = target.clone()  # 保存原始标签
            
            # 准备恶意样本标志
            attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
            
            # 前向传播
            bottom_outputs = []
            
            # 不管是否后门测试，先获取所有特征并评估推断准确率
            for i, model in enumerate(bottom_models):
                if i == args.bkd_adversary:
                    # 获取特征，用于推断
                    clean_output = model(data)
                    
                    # 对所有样本评估标签推断准确率
                    if model.label_inference and model.label_inference.initialized:
                        # 尝试推断标签
                        original_data = data.view(data.size(0), -1)
                        inferred_labels, _ = model.label_inference.infer_labels(original_data)
                        
                        if inferred_labels is not None:
                            # 计算推断准确率 - 关注推断对目标类的检测能力
                            for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                                is_target_class = (true == target_class)
                                
                                # 判断预测是否正确：
                                # - 真实是目标类且预测为目标类(pred == target_class)
                                # - 真实不是目标类且预测不是目标类(pred != target_class)
                                if (is_target_class and pred == target_class) or (not is_target_class and pred != target_class):
                                    inference_correct += 1
                                
                                inference_samples += 1
                    
                    # 如果是后门测试，准备攻击样本
                    if is_backdoor and epoch >= args.Ebkd:
                        # 将所有样本标记为攻击样本
                        attack_flags = torch.ones(batch_size, dtype=torch.bool).to(DEVICE)
                        
                        # 记录原始类别，用于类别级别ASR分析
                        for idx, orig_label in enumerate(original_target):
                            class_total[orig_label.item()] += 1
                        
                        # 更新攻击总样本数
                        attack_total += batch_size
                        
                        # 更新目标标签
                        target = torch.ones_like(target) * target_class
                        
                        # 应用触发器
                        if hasattr(model, 'label_inference') and model.label_inference.initialized:
                            output = model(data, attack_flags=attack_flags, inferred_labels=inferred_labels)
                        else:
                            output = model(data, attack_flags=attack_flags)
                    else:
                        output = clean_output
                else:
                    # 对其他模型正常处理
                    output = model(data)
                
                bottom_outputs.append(output)
            
            # 拼接底部模型输出
            combined_output = torch.cat(bottom_outputs, dim=1)
            
            # 顶部模型前向传播
            output = modelC(combined_output)
            
            # 计算损失
            test_loss += criterion(output, target).item() * batch_size
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 如果是后门测试，计算ASR
            if is_backdoor:
                pred_flat = pred.squeeze()
                # 计算总体ASR - 所有被攻击样本中有多少成功被分类为目标类
                attack_success += (pred_flat == target_class).sum().item()
                
                # 计算每个类别的ASR
                for idx, (orig_label, pred_label) in enumerate(zip(original_target, pred_flat)):
                    orig_label_item = orig_label.item()
                    if pred_label == target_class:
                        class_success[orig_label_item] += 1
                
                # 计算攻击样本预测的置信度
                with torch.no_grad():
                    probs = F.softmax(output, dim=1)
                    target_probs = probs[:, target_class]
                    asr_confidence.extend(target_probs.cpu().numpy())
                
                # 更新进度条
                current_asr = 100. * attack_success / attack_total if attack_total > 0 else 0.0
                current_inf_acc = 0.0
                if inference_samples > 0:
                    current_inf_acc = 100. * inference_correct / inference_samples
                
                pbar.set_postfix({
                    'ASR': f"{current_asr:.2f}%", 
                    'Inf_Acc': f"{current_inf_acc:.2f}%"
                })
            else:
                # 更新干净测试进度条
                current_acc = 100. * correct / ((pbar.n + 1) * batch_size)
                current_inf_acc = 0.0
                if inference_samples > 0:
                    current_inf_acc = 100. * inference_correct / inference_samples
                
                pbar.set_postfix({
                    'Acc': f"{current_acc:.2f}%",
                    'Inf_Acc': f"{current_inf_acc:.2f}%"
                })
    
    # 计算平均损失和准确率
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 计算标签推断准确率
    if inference_samples > 0:
        inference_accuracy = 100. * inference_correct / inference_samples
    else:
        inference_accuracy = 0.0
    
    # 计算ASR
    if is_backdoor and attack_total > 0:
        true_asr = 100. * attack_success / attack_total
        
        # 计算类别级别的ASR
        class_asr = {}
        for c in range(args.num_classes):
            if class_total[c] > 0:
                class_asr[c] = 100. * class_success[c] / class_total[c]
            else:
                class_asr[c] = 0.0
        
        # 计算平均攻击置信度
        avg_confidence = np.mean(asr_confidence) * 100 if asr_confidence else 0
        
        # 输出详细的ASR信息
        logger.info(f"\n{'='*20} 攻击成功率评估 {'='*20}")
        logger.info(f"全局ASR: {true_asr:.2f}% ({attack_success}/{attack_total})")
        logger.info(f"平均攻击置信度: {avg_confidence:.2f}%")
        logger.info("类别级别ASR:")
        
        for c in range(args.num_classes):
            if class_total[c] > 0:
                logger.info(f"  类别 {c}: {class_asr[c]:.2f}% ({class_success[c]}/{class_total[c]})")
        
        # 更新触发器模块的ASR统计
        if hasattr(adversary_model, 'villain_trigger'):
            adversary_model.villain_trigger.update_inference_stats(inference_accuracy, true_asr)
        
        logger.info(f"ASR更新: 当前估计 = {true_asr:.2f}%")
    else:
        true_asr = 0.0
    
    # 输出推断准确率信息
    if inference_samples > 0:
        logger.info(f"测试推断统计: 正确样本={inference_correct}/{inference_samples}, 准确率={inference_accuracy:.2f}%")
    
    return test_loss, accuracy, inference_accuracy, true_asr

def load_cinic10_dataset(data_dir, batch_size):
    """加载CINIC10数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 CINIC10 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404],
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404],
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    
    print("\n2. 检查CINIC10数据集路径...")
    data_root = data_dir
    
    # 检查CINIC10目录是否存在
    cinic10_dir = os.path.join(data_root, 'CINIC10')
    if not os.path.exists(cinic10_dir):
        print(f"错误: CINIC10数据集目录不存在: {cinic10_dir}")
        print("请确保已下载并解压CINIC10数据集。")
        sys.exit(1)
    
    print("\n3. 加载CINIC10数据集...")
    try:
        train_dataset = CINIC10Dataset(
            root=data_root,
            split='train',
            transform=transform_train
        )
        
        # CINIC10同时使用原始验证集和测试集进行测试
        test_dataset = CINIC10Dataset(
            root=data_root,
            split='test',
            transform=transform_test
        )
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("请检查数据集路径是否正确，以及是否有读取权限")
        sys.exit(1)
    
    print("\n4. 创建数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\n数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"类别数量: 10")
    
    return train_loader, test_loader

def create_models():
    """创建模型"""
    output_dim = 64
    
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
            model = CINICBottomModel(
                input_dim=output_dim,
                output_dim=output_dim,
                is_adversary=True,
                args=args
            )
            # 设置标签推断模块
            model.label_inference = label_inference
        else:
            # 创建正常模型
            model = CINICBottomModel(
                input_dim=output_dim,
                output_dim=output_dim
            )
        model.to(DEVICE)
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = CINICTopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)
    
    # 创建并设置VILLAIN触发器
    villain_trigger = VILLAINTrigger(args)
    bottom_models[args.bkd_adversary].set_villain_trigger(villain_trigger)
    
    return bottom_models, modelC

def save_checkpoint(modelC, bottom_models, optimizers, optimizer_top, epoch, clean_acc, asr=None, inference_acc=None):
    """保存模型检查点，增强版支持更全面的模型状态记录"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 生成唯一的模型名称
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp = 'ALL' if not hasattr(args, 'defense_type') or args.defense_type != 'DPSGD' else 'DPSGD'
    label_knowledge = "True" if hasattr(args, 'has_label_knowledge') and args.has_label_knowledge else "False"
    
    if asr is None:
        model_name = f"CINIC10_Clean_{temp}_{label_knowledge}_{args.party_num}"
    else:
        model_name = f"CINIC10_VILLAIN_WithInference_{temp}_{label_knowledge}_{args.party_num}"
        
        # 将关键参数添加到模型名中，方便识别不同配置
        if hasattr(args, 'advanced_trigger') and args.advanced_trigger:
            model_name += "_AdvTrigger"
        
        if hasattr(args, 'balance_metrics') and args.balance_metrics:
            model_name += "_Balance"
        
        if hasattr(args, 'adaptive_budget') and args.adaptive_budget:
            model_name += "_Adaptive"
            
    model_name += f"_E{epoch}_Acc{clean_acc:.2f}"
    
    if inference_acc is not None:
        model_name += f"_InfAcc{inference_acc:.2f}"
    
    if asr is not None:
        model_name += f"_ASR{asr:.2f}"
    
    model_name += f"_{timestamp}"
    
    checkpoint = {
        'epoch': epoch,
        'args': vars(args),
        'model_c_state_dict': modelC.state_dict() if modelC is not None else None,
        'bottom_models_state_dict': [model.state_dict() for model in bottom_models] if bottom_models else None,
        'optimizers_state_dict': [opt.state_dict() for opt in optimizers] if optimizers else None,
        'optimizer_top_state_dict': optimizer_top.state_dict() if optimizer_top is not None else None,
        'clean_acc': clean_acc,
        'inference_acc': inference_acc,
        'asr': asr,
        'timestamp': timestamp
    }
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # 保存日志文件
    log_path = os.path.join(args.checkpoint_dir, f"{model_name}_log.txt")
    with open(log_path, 'w') as f:
        # 写入基本信息
        f.write(f"模型名称: {model_name}\n")
        f.write(f"保存时间: {timestamp}\n")
        f.write(f"轮次: {epoch}\n")
        f.write(f"干净准确率: {clean_acc:.2f}%\n")
        if inference_acc is not None:
            f.write(f"推断准确率: {inference_acc:.2f}%\n")
        if asr is not None:
            f.write(f"攻击成功率: {asr:.2f}%\n")
        f.write("\n参数配置:\n")
        
        # 写入参数配置
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"模型已保存到 {checkpoint_path}")
    logger.info(f"日志已保存到 {log_path}")

def collect_inference_data(modelC, bottom_models, train_loader, args):
    """收集标签推断数据，增强版支持更高效的数据收集和平衡"""
    logger.info("\n" + "="*80)
    logger.info("收集标签推断数据并构建类别统计信息...")
    logger.info("="*80)
    
    # 确保模型处于评估模式
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 计算每个类别的目标样本量
    target_per_class = args.history_size // args.num_classes
    logger.info(f"每个类别的目标样本量: {target_per_class}")
    
    # 初始化类别计数
    class_counts = {c: 0 for c in range(args.num_classes)}
    
    # 创建进度跟踪变量
    total_batches = len(train_loader)
    max_collection_batches = min(total_batches, 200)  # 最多使用200个批次收集数据
    batches_without_progress = 0
    max_batches_without_progress = 20  # 连续20个批次没有新样本就提前结束
    
    # 收集样本过程
    with torch.no_grad():
        pbar = tqdm(train_loader, total=max_collection_batches, desc="收集推断数据")
        for batch_idx, (data, target) in enumerate(pbar):
            # 如果达到最大批次或连续多批次没有收集到新样本，提前结束
            if batch_idx >= max_collection_batches or batches_without_progress >= max_batches_without_progress:
                break
                
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 获取每个类别的当前样本数量
            prev_counts = adversary_model.label_inference.get_samples_per_class()
            
            # 前向传播获取特征表示
            for i, model in enumerate(bottom_models):
                output = model(data)
                if i == args.bkd_adversary:
                    # 更新标签推断历史
                    added = adversary_model.label_inference.update_history(output, target)
                    
                    # 检查是否有新样本添加
                    if added == 0:
                        batches_without_progress += 1
                    else:
                        batches_without_progress = 0
                        
            # 更新进度条描述
            current_counts = adversary_model.label_inference.get_samples_per_class()
            total_samples = sum(current_counts.values())
            
            # 计算完成百分比
            completion_percent = min(100, int(total_samples / (target_per_class * args.num_classes) * 100))
            
            # 更新进度条
            pbar.set_postfix({
                'samples': total_samples, 
                'progress': f"{completion_percent}%",
                'no_progress': batches_without_progress
            })
            
            # 显示更详细的进度，每10个批次
            if batch_idx % 10 == 0:
                # 找出样本较少的类别
                low_sample_classes = [c for c, count in current_counts.items() if count < target_per_class * 0.5]
                if low_sample_classes:
                    low_sample_str = ", ".join([f"类别{c}: {current_counts.get(c, 0)}" for c in low_sample_classes])
                    logger.info(f"样本不足的类别: {low_sample_str}")
            
            # 检查是否所有类别都已收集足够样本
            if all(count >= target_per_class for count in current_counts.values()):
                logger.info("\n所有类别已收集足够样本，提前结束数据收集")
                break
    
    # 完成数据收集后强制更新类别统计
    logger.info("\n数据收集完成，更新类别统计")
    success = adversary_model.label_inference.update_class_stats(force=True)
    
    # 如果成功，更新触发器状态
    if success and hasattr(adversary_model, 'villain_trigger'):
        logger.info("更新触发器状态...")
        adversary_model.villain_trigger.update_inference_stats()
        adversary_model.villain_trigger.is_initialized = True
    
    # 打印统计信息
    total_samples = adversary_model.label_inference.get_total_samples()
    class_counts = adversary_model.label_inference.get_samples_per_class()
    
    logger.info("\n" + "="*60)
    logger.info(f"标签推断数据收集完成！")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"类别分布: {class_counts}")
    logger.info(f"统计更新状态: {'成功' if success else '失败'}")
    
    # 打印类别统计信息摘要
    if success:
        logger.info("\n类别统计信息摘要:")
        
        # 检查类别均衡性
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info(f"类别不平衡比率: {imbalance_ratio:.2f} (越接近1越平衡)")
        
        for c in range(args.num_classes):
            if c in adversary_model.label_inference.means:
                mean_norm = np.linalg.norm(adversary_model.label_inference.means[c])
                count = len(adversary_model.label_inference.data_by_class[c])
                logger.info(f"  类别 {c}: 样本数={count}, 均值范数={mean_norm:.4f}")
        
        # 如果初始化了分类器，显示分类器信息
        if hasattr(adversary_model.label_inference, 'classifier') and adversary_model.label_inference.classifier is not None:
            logger.info(f"\n分类器信息:")
            train_acc = adversary_model.label_inference.accuracy_history[-1] if adversary_model.label_inference.accuracy_history else 0
            logger.info(f"  训练准确率: {train_acc:.2f}%")
    
    logger.info("="*60)
    
    return success

def prepare_backdoor_data(data, target, args):
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

def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info(f"VILLAIN 攻击训练 (带标签推断) - 数据集: CINIC10")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"参与方数量: {args.party_num}")
    logger.info(f"恶意方ID: {args.bkd_adversary}")
    logger.info(f"目标类别: {args.target_class}")
    logger.info(f"触发器大小: {args.trigger_size}")
    logger.info(f"触发器强度: {args.trigger_magnitude}")
    logger.info(f"毒化预算: {args.poison_budget}")
    if args.early_stopping:
        logger.info(f"早停: 启用 (耐心轮数={args.patience}, 监控指标={args.monitor})")
    logger.info("="*80 + "\n")
    
    # 加载数据集
    train_loader, test_loader = load_cinic10_dataset(args.data_dir, args.batch_size)
    
    # 创建模型
    bottom_models, modelC = create_models()
    
    # 打印模型结构
    logger.info("\n模型结构:")
    logger.info(f"底部模型数量: {args.party_num}")
    logger.info(f"恶意模型ID: {args.bkd_adversary}")
    for i, model in enumerate(bottom_models):
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"底部模型 {i}: {total_params:,} 参数" + (" (恶意)" if i == args.bkd_adversary else ""))
    
    total_params = sum(p.numel() for p in modelC.parameters())
    logger.info(f"顶部模型: {total_params:,} 参数")
    
    # 创建优化器
    optimizers = [optim.SGD(model.parameters(), 
                           lr=args.lr, 
                           momentum=args.momentum, 
                           weight_decay=args.weight_decay) 
                 for model in bottom_models]
    
    optimizerC = optim.SGD(modelC.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    
    # 为模型添加优化器引用（用于梯度收集）
    for i, model in enumerate(bottom_models):
        model.optimizer = optimizers[i]
    
    # 学习率调度器
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
    
    # 预训练阶段: 收集标签推断数据并强制初始化推断模块
    logger.info("\n" + "="*60)
    logger.info("预训练阶段: 收集标签推断数据")
    logger.info("="*60)
    
    # 获取恶意模型
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 确保恶意模型有触发器
    if not hasattr(adversary_model, 'villain_trigger') or adversary_model.villain_trigger is None:
        logger.info("\n警告: 恶意模型触发器未正确设置，尝试重新设置...")
        villain_trigger = VILLAINTrigger(args)
        adversary_model.set_villain_trigger(villain_trigger)
    
    # 确保推断模块得到初始化
    success = collect_inference_data(modelC, bottom_models, train_loader, args)
    
    # 显示训练配置
    logger.info("\n" + "="*40)
    logger.info("开始训练")
    logger.info(f"轮次: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"优化器: SGD (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
    logger.info(f"后门启动轮次: {args.Ebkd}")
    logger.info(f"标签推断状态: {'已初始化' if adversary_model.label_inference.initialized else '未初始化'}")
    logger.info(f"触发器状态: {'已初始化' if (hasattr(adversary_model, 'villain_trigger') and adversary_model.villain_trigger.is_initialized) else '未初始化'}")
    logger.info("="*40 + "\n")
    
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
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_asr': [],
        'train_inf_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_asr': [],
        'test_inf_acc': []
    }
    
    no_improvement_count = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        # 训练一个epoch
        train_loss, train_acc, train_inference_acc, train_asr = train_epoch(
            modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args
        )
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_asr'].append(train_asr)
        history['train_inf_acc'].append(train_inference_acc)
        
        # 测试
        test_loss, test_acc, test_inference_acc, _ = test(
            modelC, bottom_models, test_loader, is_backdoor=False, epoch=epoch, args=args
        )
        
        # 后门测试
        bkd_loss, bkd_acc, bkd_inference_acc, true_asr = test(
            modelC, bottom_models, test_loader, is_backdoor=True, epoch=epoch, args=args
        )
        
        # 记录测试历史
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_asr'].append(true_asr)
        history['test_inf_acc'].append(test_inference_acc)
        
        # 更新学习率
        for scheduler in schedulers:
            scheduler.step(test_acc)
        schedulerC.step(test_acc)
        
        # 打印训练信息
        logger.info(f"\nEpoch {epoch} 结果:")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train ASR: {train_asr:.2f}%, Train Inference Acc: {train_inference_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Inference Acc: {test_inference_acc:.2f}%")
        logger.info(f"Backdoor Loss: {bkd_loss:.4f}, Backdoor Acc: {bkd_acc:.2f}%, True ASR: {true_asr:.2f}%, Backdoor Inference Acc: {bkd_inference_acc:.2f}%")
        
        # 自适应预算调整 - 确保ASR不会过快达到100%
        if args.adaptive_budget and true_asr > 95.0 and epoch >= args.Ebkd + 3:
            new_budget = max(0.05, args.poison_budget * 0.8)
            if new_budget != args.poison_budget:
                logger.info(f"\n降低毒化预算: {args.poison_budget:.2f} -> {new_budget:.2f} (ASR={true_asr:.1f}%)")
                args.poison_budget = new_budget
                
        # 强制进行标签推断更新 (每5个epoch)
        if epoch % 5 == 0 and adversary_model.label_inference is not None:
            logger.info("\n强制更新标签推断...")
            # 强制更新类别统计信息
            if hasattr(adversary_model.label_inference, 'update_class_stats'):
                adversary_model.label_inference.update_class_stats(force=True)
            
            # 打印当前状态
            if adversary_model.label_inference.initialized:
                logger.info(f"标签推断状态: 已初始化")
                if hasattr(adversary_model.label_inference, 'get_total_samples'):
                    logger.info(f"梯度历史大小: {adversary_model.label_inference.get_total_samples()}")
                
                # 更新触发器状态
                if hasattr(adversary_model, 'villain_trigger'):
                    adversary_model.villain_trigger.update_inference_stats(test_inference_acc, true_asr)
                    
                    # 如果ASR已经接近100%，减小触发器强度
                    if true_asr > 95.0 and test_acc < 60.0:
                        old_magnitude = adversary_model.villain_trigger.trigger_magnitude
                        adversary_model.villain_trigger.trigger_magnitude = max(0.5, old_magnitude * 0.8)
                        logger.info(f"触发器强度过高，ASR={true_asr:.1f}%, 干净准确率={test_acc:.1f}%")
                        logger.info(f"减小触发器强度: {old_magnitude:.2f} -> {adversary_model.villain_trigger.trigger_magnitude:.2f}")
        
        # 早停逻辑 - 修复为组合权重策略
        # 只有在后门攻击真正开始后才保存模型 (ASR > 5%)
        should_evaluate = true_asr > 5.0 or epoch == 1
        
        if should_evaluate:
            # 使用clean accuracy和ASR的组合权重 (各占50%)
            combined_score = 0.5 * test_acc + 0.5 * true_asr
            best_combined_score = 0.5 * best_accuracy + 0.5 * best_asr
            
            print(f"组合评分: 当前={combined_score:.2f} (0.5*{test_acc:.1f}% + 0.5*{true_asr:.1f}%), 最佳={best_combined_score:.2f}")
            
            # 检查是否需要保存新的最佳模型
            is_improvement = combined_score > best_combined_score
            
            # 为第一个有效epoch设置特殊处理
            if epoch == 1 and true_asr == 0:
                is_improvement = False  # 不保存ASR为0的模型
                print("跳过保存: 等待后门攻击开始 (ASR > 5%)")
            
            if is_improvement:
                # 更新所有最佳指标
                best_accuracy = test_acc
                best_inference_acc = test_inference_acc
                best_asr = true_asr
                best_epoch = epoch
                
                # 更新指标字典
                best_metrics = {
                    'test_acc': test_acc,
                    'inference_acc': test_inference_acc,
                    'asr': true_asr,
                    'combined_score': combined_score,
                    'epoch': epoch
                }
                    
                # 保存最佳模型
                save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr, test_inference_acc)
                logger.info(f"\n保存最佳模型 (Epoch {epoch}) - 组合评分提升")
                logger.info(f"Clean Acc: {test_acc:.2f}%, ASR: {true_asr:.2f}%, 组合评分: {combined_score:.2f}")
                
                # 重置早停计数器
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                logger.info(f"\n没有改进: {no_improvement_count}/{args.patience} (最佳组合评分: {best_combined_score:.2f} 在 Epoch {best_epoch})")
                
                # 检查是否达到早停条件
                if args.early_stopping and no_improvement_count >= args.patience:
                    logger.info(f"\n早停触发! {args.patience} 轮次内组合评分没有改进。")
                    logger.info(f"最佳模型 (Epoch {best_epoch}):")
                    logger.info(f"Clean Acc: {best_accuracy:.2f}%, ASR: {best_asr:.2f}%, 组合评分: {best_combined_score:.2f}")
                    break
        else:
            logger.info(f"等待后门攻击开始 (当前ASR: {true_asr:.2f}% < 5%)")
            # 重置early stopping计数，因为还没有有效的评估
            no_improvement_count = 0
    
    # 训练结束，输出详细的最佳结果
    logger.info("\n" + "="*60)
    logger.info(f"训练完成！最佳模型 (Epoch {best_metrics['epoch']}):")
    logger.info(f"Clean Accuracy: {best_metrics['test_acc']:.2f}%")
    logger.info(f"Inference Accuracy: {best_metrics['inference_acc']:.2f}%")
    logger.info(f"Attack Success Rate: {best_metrics['asr']:.2f}%")
    logger.info("="*60)
    
    # 保存最终配置和结果
    results = {
        'args': vars(args),
        'best_metrics': best_metrics,
        'training_history': history
    }
    
    try:
        import json
        with open(os.path.join(args.checkpoint_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"训练结果和配置已保存到 {os.path.join(args.checkpoint_dir, 'training_results.json')}")
    except Exception as e:
        logger.warning(f"保存结果时出错: {str(e)}")
    
    logger.info("\n使用早停的运行命令示例:")
    logger.info(f"python train_cinic_villain_with_inference.py --data-dir '/path/to/data' --batch-size 32 --epochs 100 "
              f"--early-stopping --patience 15 --monitor asr --trigger-magnitude 3.0 --adaptive-threshold "
              f"--feature-selection --advanced-trigger --gpu 0")
    
    return best_metrics

if __name__ == '__main__':
    main() 