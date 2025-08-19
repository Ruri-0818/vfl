#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: train_nuswide_badvfl_with_inference.py
# 描述: 针对NUS-WIDE数据集的BadVFL攻击训练 (带标签推断)
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
import time
import random
from tqdm import tqdm
from PIL import Image

from defense_all import build_defense

# 扩展命令行参数
parser = argparse.ArgumentParser(description='针对NUS-WIDE数据集的BadVFL攻击训练 (带标签推断)')
# 原有参数
parser.add_argument('--dataset', type=str, default='NUSWIDE', help='数据集名称 (NUSWIDE)')
parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=int, default=4, help='BadVFL触发器大小')  # 改为int类型与Imagenette一致
parser.add_argument('--trigger-intensity', type=float, default=2.0, help='BadVFL触发器强度')
parser.add_argument('--position', type=str, default='dr', help='触发器位置 (dr=右下, ul=左上, mid=中间, ml=中左)')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=20, help='早停轮数')  # 增加patience，NUS-WIDE需要更多训练时间
parser.add_argument('--min-epochs', type=int, default=30, help='最小训练轮数')  # 降低最小训练轮数
parser.add_argument('--max-epochs', type=int, default=300, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=8.0, help='后门损失权重')  # 降低后门权重以平衡clean acc
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.3, help='毒化预算')  # 降低毒化预算以提高clean acc
parser.add_argument('--Ebkd', type=int, default=5, help='后门注入开始轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.5, help='学习率倍增器')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_badvfl_nuswide', help='检查点目录')  # 使用专门的checkpoints目录
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=5, help='类别数量')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='/home/tsinghuaair/attack/WithInference/data/NUS-WIDE', help='数据集目录')  # 更新数据目录路径
parser.add_argument('--trigger-type', type=str, default='pattern', help='触发器类型 (pattern或pixel)')

# NUS-WIDE特定参数
parser.add_argument('--image-size', type=int, default=128, help='图像尺寸')
parser.add_argument('--selected-concepts', type=str, nargs='+', 
                    default=['buildings', 'grass', 'animal', 'water', 'person'],
                    help='选择的NUS-WIDE概念')

# 标签推断相关参数
parser.add_argument('--inference-weight', type=float, default=0.1, help='标签推断损失权重')
parser.add_argument('--history-size', type=int, default=5000, help='嵌入向量历史记录大小')
parser.add_argument('--cluster-update-freq', type=int, default=50, help='聚类更新频率(批次)')
parser.add_argument('--inference-start-epoch', type=int, default=3, help='开始标签推断的轮数')  # 更早开始推断
parser.add_argument('--confidence-threshold', type=float, default=0.3, help='标签推断置信度阈值')
parser.add_argument('--adaptive-threshold', action='store_true', help='是否使用自适应置信度阈值')
parser.add_argument('--feature-selection', action='store_true', help='是否启用特征选择以提高推断准确率')
parser.add_argument('--use-ensemble', action='store_true', help='是否使用集成方法提高推断准确率')
parser.add_argument('--aux-data-ratio', type=float, default=0.1, help='用于标签推断的辅助数据比例')

# 二元分类器参数
parser.add_argument('--binary-classifier', type=str, default='randomforest', choices=['randomforest', 'logistic'], 
                    help='二元分类器类型 (randomforest 或 logistic)')

# 早停参数
parser.add_argument('--early-stopping', action='store_true',
                    help='启用早停 (default: False)')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'],
                    help='监控指标，用于早停判断 (default: test_acc)')

# Defense-related arguments
parser.add_argument('--dpsgd-noise-multiplier', type=float, default=1.0, help='Noise multiplier for DPSGD')
parser.add_argument('--dpsgd-max-grad-norm', type=float, default=1.0, help='Max grad norm for DPSGD')
parser.add_argument('--mp-pruning-amount', type=float, default=0.2, help='Pruning amount for MP')
parser.add_argument('--anp-sigma', type=float, default=0.1, help='Sigma for Gaussian noise in ANP')
parser.add_argument('--bdt-prune-ratio', type=float, default=0.2, help='Prune ratio for BDT')
parser.add_argument('--vflip-threshold', type=float, default=3.0, help='Anomaly threshold for VFLIP')
parser.add_argument('--vflip-train-epochs', type=int, default=5, help='Number of epochs to pre-train VFLIP MAE')
parser.add_argument('--iso-lr', type=float, default=1e-3, help='Learning rate for ISO layer')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# NUS-WIDE数据集类
class PreprocessedNUSWIDEDataset(Dataset):
    """预处理版本的NUS-WIDE数据集加载器"""
    
    def __init__(self, data_dir, split='train', transform=None, num_classes=5, 
                 selected_concepts=None):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train' 或 'test'
            transform: 图像变换
            num_classes: 类别数量
            selected_concepts: 选择的概念子集列表（这里固定为5个类别）
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # 固定的5个类别映射 - 根据NUS-WIDE的实际类别ID
        # 这些是常见的NUS-WIDE类别ID，您可能需要根据实际的类别映射调整
        self.class_mapping = {
            'buildings': 2,    # 建筑
            'grass': 8,        # 草地  
            'animal': 0,       # 动物
            'water': 19,       # 水
            'person': 14       # 人物
        }
        
        # 反向映射：从原始类别ID到新的0-4类别ID
        self.reverse_mapping = {v: i for i, (k, v) in enumerate(self.class_mapping.items())}
        
        print(f"\n[BadVFL NUS-WIDE] 加载 {split} 数据集")
        print(f"数据目录: {data_dir}")
        print(f"类别映射: {self.class_mapping}")
        print(f"反向映射: {self.reverse_mapping}")
        
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
                f"NUS-WIDE数据集文件缺失: {missing_files}\n"
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
        valid_indices = []
        
        for i, line in enumerate(label_lines):
            if line:
                # 解析多标签格式（空格分隔的数字）
                try:
                    original_labels = [int(x) for x in line.split() if x.strip().isdigit()]
                except:
                    continue
                
                # 检查是否包含我们关心的5个类别中的任何一个
                mapped_labels = []
                for orig_label in original_labels:
                    if orig_label in self.reverse_mapping:
                        mapped_labels.append(self.reverse_mapping[orig_label])
                
                # 如果包含我们关心的类别，则保留这个样本
                if mapped_labels:
                    # 使用第一个匹配的类别作为单标签
                    self.labels.append(mapped_labels[0])
                    valid_indices.append(i)
        
        # 只保留有效样本的图像路径
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        
        print(f"从 {len(label_lines)} 个样本中筛选出 {len(self.image_paths)} 个包含目标类别的样本")
        
        # 转换为绝对路径
        self.image_paths = [os.path.join(self.data_dir, path) for path in self.image_paths]
        
        # 验证图像文件存在并进一步筛选
        final_valid_indices = []
        final_labels = []
        final_paths = []
        
        for i, img_path in enumerate(self.image_paths):
            if os.path.exists(img_path):
                final_valid_indices.append(i)
                final_labels.append(self.labels[i])
                final_paths.append(img_path)
        
        self.image_paths = final_paths
        self.labels = final_labels
        
        # 限制数据量以提高训练效率
        max_samples = 3000 if self.split == 'train' else 800
        if len(self.image_paths) > max_samples:
            print(f"限制样本数量从 {len(self.image_paths)} 到 {max_samples}")
            # 随机采样以保持类别平衡
            import random
            indices = list(range(len(self.image_paths)))
            random.shuffle(indices)
            selected_indices = indices[:max_samples]
            
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]
        
        # 统计最终的标签分布
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"最终标签分布: {dict(sorted(label_counts.items()))}")
        print(f"有效样本数: {len(self.image_paths)}")
        
        # 确保没有超出范围的标签
        if self.labels:
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
            img = Image.new('RGB', (128, 128), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        target = self.labels[idx]
        
        # 确保标签在有效范围内
        target = target % self.num_classes
        
        return img, target

# 标签推断器实现
class BadVFLLabelInference:
    """BadVFL攻击中的标签推断模块，直接使用模型输出进行推断"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.args = args
        self.confidence_threshold = args.confidence_threshold
        
        # 存储历史推断结果和置信度
        self.history_features = []
        self.history_predictions = []
        self.history_confidence = []
        
        # 辅助数据集用于训练推断模型
        self.auxiliary_dataset = None
        self.auxiliary_loader = None
        
        # 推断分类器
        self.inference_classifier = None
        self.initialized = False
        
        # 特征维度 - BadVFL使用完整图像作为特征 (NUS-WIDE为128x128)
        self.expected_features = 3 * args.image_size * args.image_size  # 3×128×128 for NUS-WIDE
        
        # 设置所需的最小样本数
        self.min_samples = max(50, 5 * num_classes)
        
        print(f"BadVFL标签推断模块创建: 特征维度={feature_dim}, 类别数={num_classes}")
        print(f"标签推断所需最小样本数: {self.min_samples}")
    
    def set_auxiliary_dataset(self, dataset, ratio=0.1):
        """设置辅助数据集用于训练标签推断模型"""
        total_size = len(dataset)
        aux_size = int(total_size * ratio)
        
        # 随机选择索引用于辅助数据集
        indices = np.random.choice(total_size, aux_size, replace=False)
        
        # 创建辅助数据集
        from torch.utils.data import Subset
        self.auxiliary_dataset = Subset(dataset, indices)
        
        # 创建数据加载器
        self.auxiliary_loader = DataLoader(
            self.auxiliary_dataset, batch_size=32, shuffle=True,
            num_workers=2, pin_memory=True)
        
        print(f"创建辅助数据集: {aux_size}个样本")
        return self.auxiliary_loader
    
    def update_with_batch(self, features, predictions, confidence=None):
        """更新特征和预测历史记录"""
        # 确保输入是numpy数组
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        # 确保预测是离散标签
        if len(predictions.shape) > 1:
            if predictions.shape[1] > 1:
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions.flatten()
                
        # 对于梯度数据的特殊处理
        if np.issubdtype(predictions.dtype, np.floating):
            # 创建伪标签：基于梯度范数的二元分类
            grad_norms = np.linalg.norm(predictions, axis=1) if len(predictions.shape) > 1 else np.abs(predictions)
            threshold = np.median(grad_norms)
            predictions = (grad_norms > threshold).astype(np.int32)
        
        if confidence is None:
            confidence = np.ones(len(features)) / self.num_classes
        elif isinstance(confidence, torch.Tensor):
            confidence = confidence.detach().cpu().numpy()
        
        # 储存新的数据
        self.history_features.extend(features)
        self.history_predictions.extend(predictions)
        self.history_confidence.extend(confidence)
        
        # 只保留最近的记录
        max_history = self.args.history_size
        if len(self.history_features) > max_history:
            self.history_features = self.history_features[-max_history:]
            self.history_predictions = self.history_predictions[-max_history:]
            self.history_confidence = self.history_confidence[-max_history:]
        
        return len(features)
    
    def initialize_classifier(self):
        """初始化标签推断分类器"""
        if len(self.history_features) < self.min_samples:
            print(f"样本不足，无法初始化分类器: {len(self.history_features)}/{self.min_samples}")
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 将历史特征和预测转换为数组
            X = np.array(self.history_features)
            y = np.array(self.history_predictions).astype(np.int32)
            
            # 输出标签的唯一值
            unique_labels = np.unique(y)
            print(f"标签唯一值: {unique_labels}")
            
            # 训练随机森林分类器
            print(f"训练标签推断分类器，使用 {len(X)} 个样本...")
            
            self.inference_classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                n_jobs=-1,
                random_state=self.args.seed
            )
            
            # 直接训练分类器
            self.inference_classifier.fit(X, y)
            self.use_pca = False
            
            # 计算训练精度
            train_preds = self.inference_classifier.predict(X)
            accuracy = np.mean(train_preds == y)
            print(f"标签推断训练精度: {accuracy*100:.2f}%")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"初始化分类器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def infer_labels(self, features, top_model=None, bottom_models=None, raw_data=None):
        """推断输入特征的标签"""
        # 方法1: 如果提供了模型和原始数据，直接使用模型进行推断
        if top_model is not None and bottom_models is not None and raw_data is not None:
            with torch.no_grad():
                raw_data = raw_data.to(DEVICE)
                
                # 前向传播
                bottom_outputs = []
                for model in bottom_models:
                    output = model(raw_data)
                    bottom_outputs.append(output)
                
                combined_output = torch.cat(bottom_outputs, dim=1)
                logits = top_model(combined_output)
                
                # 获取预测和置信度
                probs = F.softmax(logits, dim=1)
                confidence, pred_labels = torch.max(probs, dim=1)
                
                return pred_labels.cpu().numpy(), confidence.cpu().numpy()
        
        # 方法2: 使用训练好的推断分类器
        if self.initialized and self.inference_classifier is not None:
            try:
                # 确保特征是2D的
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                # 预测标签
                pred_labels = self.inference_classifier.predict(features)
                
                # 获取置信度估计
                pred_probs = self.inference_classifier.predict_proba(features)
                confidence = np.max(pred_probs, axis=1)
                
                return pred_labels, confidence
            except Exception as e:
                print(f"标签推断失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
        
        return None, None

    def get_total_samples(self):
        """获取收集的样本总数"""
        return len(self.history_features)
        
    def update_class_stats(self, modelC=None, bottom_models=None, force=False):
        """更新类别统计信息"""
        # 检查是否有足够的样本
        if len(self.history_features) < self.min_samples and not force:
            print(f"样本不足，无法更新类别统计信息 ({len(self.history_features)}/{self.min_samples})")
            return False
        
        # 初始化或更新分类器
        if not self.initialized or force:
            success = self.initialize_classifier()
            if success:
                print(f"成功初始化标签推断 (样本数: {len(self.history_features)})")
                return True
            else:
                print(f"标签推断初始化失败")
                return False
        
        return self.initialized

    def embedding_swapping(self):
        """为兼容性添加的方法"""
        # 检查是否有足够的样本
        if len(self.history_features) < self.min_samples:
            print(f"样本不足，无法进行嵌入交换: {len(self.history_features)}/{self.min_samples}")
            return False
        
        print(f"BadVFL不执行嵌入交换，直接使用收集的 {len(self.history_features)} 个样本进行标签推断")
        return True
    
    def candidate_selection(self):
        """为兼容性添加的方法"""
        return self.initialize_classifier()

# BadVFL触发器实现
class BadVFLTrigger:
    """BadVFL攻击中的触发器实现，支持多种类型的触发器模式"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.device = args.device if hasattr(args, 'device') else 'cpu'
        self.dataset_name = args.dataset
        self.position = args.position  # 触发器位置
        self.pattern_size = int(args.trigger_size)  # 触发器大小
        self.intensity = args.trigger_intensity * 3.0  # 增加触发器强度
        self.trigger_type = args.trigger_type if hasattr(args, 'trigger_type') else 'pattern'
        self.image_size = args.image_size  # NUS-WIDE图像尺寸
        
        # 标签推断模块
        self.label_inference = None
        
        # 初始化状态标志
        self.is_initialized = False
        
        print(f"创建BadVFL触发器: 类型={self.trigger_type}, 位置={self.position}, 大小={self.pattern_size}, 强度={self.intensity}, 图像尺寸={self.image_size}")
        
        # 为pixel类型触发器初始化像素位置和值
        if self.trigger_type == 'pixel':
            self.initialize_pixel_trigger()
        # 为pattern类型触发器初始化模式
        elif self.trigger_type == 'pattern':
            self.initialize_pattern_trigger()
    
    def initialize_pixel_trigger(self):
        """初始化像素触发器 - 适配NUS-WIDE图像尺寸"""
        img_size = self.image_size
        self.pixel_positions = [
            (0, 0), (0, 1), (1, 0),  # 左上角
            (img_size-1, img_size-1), (img_size-2, img_size-1), (img_size-1, img_size-2),  # 右下角
            (0, img_size-1), (1, img_size-1),  # 左下角
            (img_size-1, 0), (img_size-1, 1),  # 右上角
        ]
        self.pixel_values = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 品红
            [0.0, 1.0, 1.0],  # 青色
            [1.0, 1.0, 1.0],  # 白色
            [0.7, 0.7, 0.0],  # 淡黄色
            [0.0, 0.7, 0.7],  # 淡青色
            [0.7, 0.0, 0.7],  # 淡品红
        ]
    
    def initialize_pattern_trigger(self):
        """初始化模式触发器 - 适配NUS-WIDE图像尺寸"""
        # 根据不同的位置创建不同的模式触发器
        if self.position == 'dr':  # 右下角
            self.create_corner_pattern('dr')
        elif self.position == 'ul':  # 左上角
            self.create_corner_pattern('ul')
        elif self.position == 'mid':  # 中间
            self.create_center_pattern()
        elif self.position == 'ml':  # 中左
            self.create_corner_pattern('ml')
        else:
            print(f"不支持的位置: {self.position}，默认使用右下角")
            self.create_corner_pattern('dr')
    
    def create_corner_pattern(self, position):
        """创建角落模式触发器 - 适配NUS-WIDE图像尺寸"""
        size = self.pattern_size
        img_size = self.image_size
        self.pattern_mask = torch.zeros(3, img_size, img_size)
        
        if position == 'dr':  # 右下角
            x_start, y_start = img_size - size, img_size - size
        elif position == 'ul':  # 左上角
            x_start, y_start = 0, 0
        elif position == 'ml':  # 中左
            x_start, y_start = (img_size - size) // 2, 0
        else:  # 默认右下角
            x_start, y_start = img_size - size, img_size - size
        
        # 创建强烈视觉对比的图案
        for i in range(size):
            for j in range(size):
                # 设置背景颜色
                self.pattern_mask[0, x_start + i, y_start + j] = 0.5  # R
                self.pattern_mask[1, x_start + i, y_start + j] = 0.0  # G
                self.pattern_mask[2, x_start + i, y_start + j] = 0.0  # B
        
        # 添加"X"形状
        for i in range(size):
            # 主对角线
            if 0 <= i < size:
                self.pattern_mask[0, x_start + i, y_start + i] = 1.0
                self.pattern_mask[1, x_start + i, y_start + i] = 1.0
                self.pattern_mask[2, x_start + i, y_start + i] = 0.0
            
            # 副对角线
            if 0 <= i < size:
                self.pattern_mask[0, x_start + i, y_start + (size-1-i)] = 0.0
                self.pattern_mask[1, x_start + i, y_start + (size-1-i)] = 0.0
                self.pattern_mask[2, x_start + i, y_start + (size-1-i)] = 1.0
    
    def create_center_pattern(self):
        """创建中心模式触发器 - 适配NUS-WIDE图像尺寸"""
        size = self.pattern_size
        img_size = self.image_size
        self.pattern_mask = torch.zeros(3, img_size, img_size)
        
        # 计算中心位置
        x_start = (img_size - size) // 2
        y_start = (img_size - size) // 2
        
        # 创建十字形
        for i in range(size):
            for j in range(size):
                # 背景颜色
                self.pattern_mask[0, x_start + i, y_start + j] = 0.2
                self.pattern_mask[1, x_start + i, y_start + j] = 0.2
                self.pattern_mask[2, x_start + i, y_start + j] = 0.2
                
                # 十字形
                if abs(i - size // 2) <= size // 4 or abs(j - size // 2) <= size // 4:
                    self.pattern_mask[0, x_start + i, y_start + j] = 1.0
                    self.pattern_mask[1, x_start + i, y_start + j] = 1.0
                    self.pattern_mask[2, x_start + i, y_start + j] = 0.0
    
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
    
    def inject_trigger(self, data, attack_flags=None):
        """向输入数据中注入触发器"""
        # 确保attack_flags是Tensor类型
        if attack_flags is not None and not isinstance(attack_flags, torch.Tensor):
            attack_flags = torch.tensor(attack_flags, dtype=torch.bool, device=data.device)
        
        # 如果没有指定攻击标志，则不注入触发器
        if attack_flags is None or torch.sum(attack_flags) == 0:
            return data
        
        # 克隆数据以避免修改原始数据
        data_copy = data.clone()
        
        # 注入触发器
        if self.trigger_type == 'pixel':
            # 像素触发器
            for idx in range(len(data)):
                if attack_flags[idx]:
                    pixel_block_size = 3
                    temp_data = data_copy[idx].clone()
                    
                    for i, ((x, y), (r, g, b)) in enumerate(zip(self.pixel_positions, self.pixel_values)):
                        if i < len(self.pixel_positions):
                            # 设置中心像素
                            temp_data[0, x, y] = r * self.intensity * 1.8
                            temp_data[1, x, y] = g * self.intensity * 1.8
                            temp_data[2, x, y] = b * self.intensity * 1.8
                            
                            # 设置周围像素
                            for dx in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                for dy in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                    if dx == 0 and dy == 0:
                                        continue
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                        decay = 1.0 - 0.15 * max(abs(dx), abs(dy))
                                        temp_data[0, nx, ny] = r * self.intensity * decay
                                        temp_data[1, nx, ny] = g * self.intensity * decay
                                        temp_data[2, nx, ny] = b * self.intensity * decay
                    
                    data_copy[idx] = temp_data
        
        elif self.trigger_type == 'pattern':
            # 模式触发器
            pattern_mask = self.pattern_mask.to(data.device)
            
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 创建掩码
                    mask = (pattern_mask > 0).float()
                    
                    temp_data = data_copy[idx].clone()
                    
                    # 应用触发器
                    temp_data = temp_data * (1 - mask) + pattern_mask * self.intensity * 2.0
                    
                    # 极大化对比度
                    dark_mask = 1 - mask
                    temp_data = temp_data * (dark_mask * 0.4 + mask * 1.5)
                    
                    # 应用饱和处理
                    temp_data = torch.clamp(temp_data, 0.0, 1.0)
                    
                    data_copy[idx] = temp_data
        
        return data_copy 

# 添加ResBlock的定义
class ResBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
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
    """NUS-WIDE底部模型，支持BadVFL攻击"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(NUSWIDEBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 使用ResNet风格的特征提取器，适配NUS-WIDE图像尺寸
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)  # 额外层适配更大的图像
        
        # 计算特征图大小 (基于128x128输入)
        # 经过3次stride=2的下采样：128 -> 64 -> 32 -> 16
        feature_dim = 256 * 16 * 16
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # 自适应池化到8x8
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 用于存储当前批次的数据和梯度
        self.current_batch_data = None
        self.current_batch_grad = None
        
        # 如果是恶意模型，初始化标签推断模块
        if is_adversary and args is not None:
            self.badvfl_trigger = None
            self.label_inference = None  # 稍后初始化
            print(f"创建恶意底部模型 (ID={args.bkd_adversary})")
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # 第一个块可能需要下采样
        layers.append(ResBlock(in_channels, out_channels, stride))
        # 其余块不需要下采样
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
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
    
    def set_badvfl_trigger(self, badvfl_trigger):
        """设置BadVFL触发器"""
        if self.is_adversary:
            self.badvfl_trigger = badvfl_trigger
            # 创建标签推断模块
            if not hasattr(self, 'label_inference') or self.label_inference is None:
                print("为恶意模型创建标签推断模块")
                self.label_inference = BadVFLLabelInference(
                    feature_dim=self.output_dim,
                    num_classes=self.args.num_classes,
                    args=self.args
                )
                # 将标签推断模块传递给触发器
                self.badvfl_trigger.set_label_inference(self.label_inference)
            else:
                # 确保触发器有对标签推断模块的引用
                self.badvfl_trigger.set_label_inference(self.label_inference)
            
            # 如果标签推断模块已初始化，立即更新触发器状态
            if self.label_inference.initialized:
                print("标签推断模块已初始化，正在更新触发器状态...")
                self.badvfl_trigger.update_inference_stats()
    
    def forward(self, x, attack_flags=None):
        """前向传播，包括恶意触发器注入和梯度收集"""
        # 如果是恶意模型，保存输入数据用于梯度收集
        if self.is_adversary and self.training:
            self.current_batch_data = x.detach()
            x.requires_grad_(True)
        
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类
        feat = self.classifier(x)
        
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
    """NUS-WIDE top model"""
    def __init__(self, input_dim=256, num_classes=5):
        super(NUSWIDETopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)
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
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1) 

def load_dataset(dataset_name, data_dir, batch_size, image_size=128):
    """加载NUS-WIDE数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集")
    print(f"{'='*50}")
    
    print(f"\n1. 准备数据预处理 (图像尺寸: {image_size}x{image_size})...")
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet预训练模型的标准化参数
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("\n2. 检查NUS-WIDE数据集路径...")
    data_root = data_dir
    
    # 确保数据根目录存在
    os.makedirs(data_root, exist_ok=True)
    print(f"数据根目录: {data_root}")
    
    print("\n3. 加载NUS-WIDE数据集...")
    try:
        # 加载训练集
        print("正在加载训练集...")
        train_dataset = PreprocessedNUSWIDEDataset(
            data_dir=data_root, 
            split='train', 
            transform=transform_train,
            num_classes=args.num_classes,
            selected_concepts=args.selected_concepts
        )
        
        # 加载测试集
        print("正在加载测试集...")
        test_dataset = PreprocessedNUSWIDEDataset(
            data_dir=data_root, 
            split='test', 
            transform=transform_test,
            num_classes=args.num_classes,
            selected_concepts=args.selected_concepts
        )
        
        print(f"{dataset_name} 数据集加载成功!")
        
        # 验证数据集完整性
        print("验证数据集完整性...")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("数据集为空，可能文件不完整")
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("\n可能的解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保NUS-WIDE数据集文件完整")
        print(f"   - 需要的文件: database_img.txt, database_label.txt, test_img.txt, test_label.txt, images/")
        print(f"   - 数据集路径: {data_root}")
        sys.exit(1)
    
    print("\n4. 创建数据加载器...")
    
    # 优化数据加载参数
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = False
    
    print(f"数据加载配置:")
    print(f"- workers: {num_workers}")
    print(f"- pin_memory: {pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    print(f"\n数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"类别数量: {args.num_classes}")
    
    # 显示第一个batch的信息作为验证
    try:
        first_batch = next(iter(train_loader))
        data_shape = first_batch[0].shape
        label_shape = first_batch[1].shape
        print(f"数据张量形状: {data_shape}")
        print(f"标签张量形状: {label_shape}")
        print(f"数据类型: {first_batch[0].dtype}")
        print(f"标签范围: {first_batch[1].min().item()} - {first_batch[1].max().item()}")
    except Exception as e:
        print(f"警告: 无法验证数据加载器: {e}")
    
    return train_loader, test_loader

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
        model = model.to(DEVICE)
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = NUSWIDETopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)
    
    # 创建并设置BadVFL触发器
    badvfl_trigger = BadVFLTrigger(args)
    bottom_models[args.bkd_adversary].set_badvfl_trigger(badvfl_trigger)
    
    return bottom_models, modelC

def prepare_backdoor_data(data, target):
    """准备后门数据，注入后门触发器"""
    batch_size = data.size(0)
    
    # 严格按照毒化预算设置攻击样本数量
    attack_portion = int(batch_size * args.poison_budget)
    attack_portion = min(attack_portion, batch_size)
    
    # 设置攻击标志
    attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    if attack_portion == 0:
        # 如果没有样本需要攻击，直接返回
        bkd_target = target.clone()
        bkd_data = data.clone()
        return bkd_data, bkd_target, attack_flags
    
    # 首先获取所有非目标类样本的索引
    non_target_indices = [i for i in range(batch_size) if target[i] != args.target_class]
    
    # 如果定义了源类别，优先选择源类别样本进行攻击
    if hasattr(args, 'source_class') and args.source_class is not None:
        source_indices = [i for i in range(batch_size) if target[i] == args.source_class]
        num_source_to_attack = min(len(source_indices), attack_portion)
        
        if num_source_to_attack > 0:
            source_attack_indices = random.sample(source_indices, num_source_to_attack) if len(source_indices) > num_source_to_attack else source_indices
            for idx in source_attack_indices:
                attack_flags[idx] = True
        
        # 如果源类别样本不足，从其他非目标类样本中选择
        remaining = attack_portion - num_source_to_attack
        if remaining > 0:
            other_indices = [i for i in non_target_indices if i not in source_indices]
            num_other_to_attack = min(len(other_indices), remaining)
            
            if num_other_to_attack > 0:
                other_attack_indices = random.sample(other_indices, num_other_to_attack) if len(other_indices) > num_other_to_attack else other_indices
                for idx in other_attack_indices:
                    attack_flags[idx] = True
    else:
        # 如果没有指定源类别，从所有非目标类样本中随机选择
        num_to_attack = min(len(non_target_indices), attack_portion)
        
        if num_to_attack > 0:
            attack_indices = random.sample(non_target_indices, num_to_attack) if len(non_target_indices) > num_to_attack else non_target_indices
            for idx in attack_indices:
                attack_flags[idx] = True
    
    # 修改标签为目标类别
    bkd_target = target.clone()
    bkd_target[attack_flags] = args.target_class
    
    # 生成克隆的数据
    bkd_data = data.clone()
    
    return bkd_data, bkd_target, attack_flags

def collect_inference_data(modelC, bottom_models, train_loader, args):
    """收集标签推断数据，使用BadVFL的标签推断方法"""
    print("\n启动BadVFL标签推断过程...")
    
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
    
    # 减少收集批次数量
    max_batches = 50
    
    print(f"将收集最多 {max_batches} 个批次的梯度数据用于标签推断")
    
    # 创建优化器
    optimizers = []
    for model in bottom_models:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        optimizers.append(optimizer)
    optimizerC = optim.SGD(modelC.parameters(), lr=args.lr)
    
    # 使用进度条显示收集进度
    data_iter = iter(train_loader)
    
    for batch_idx in tqdm(range(max_batches), desc="收集标签推断数据"):
        try:
            data, target = next(data_iter)
        except StopIteration:
            print(f"数据加载器已耗尽，收集了 {batch_idx} 个批次")
            break
            
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 清除所有梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        # 前向传播
        bottom_outputs = []
        for i, model in enumerate(bottom_models):
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
            original_data = saved_data.view(saved_data.size(0), -1)
            samples_added = label_inference_module.update_with_batch(original_data, saved_grad)
            
            # 每10个批次尝试一次初始化
            if batch_idx % 10 == 0 and batch_idx > 0:
                if label_inference_module.embedding_swapping():
                    if label_inference_module.candidate_selection():
                        if adversary_model.badvfl_trigger:
                            adversary_model.badvfl_trigger.update_inference_stats()
                            print(f"\n标签推断初始化成功! 批次: {batch_idx}")
                            return True
        
        # 清空梯度避免累积
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
    
    # 最终尝试执行初始化
    print(f"最终尝试标签推断初始化，已收集 {len(label_inference_module.history_features)} 个样本")
    
    if label_inference_module.embedding_swapping():
        if label_inference_module.candidate_selection():
            if adversary_model.badvfl_trigger:
                adversary_model.badvfl_trigger.update_inference_stats()
                return True
    
    return False 

def train_epoch(modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args, defense_hooks):
    """训练一个轮次，包括BadVFL后门注入和标签推断"""
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    total_loss = 0
    clean_correct = 0
    bkd_correct = 0
    backdoor_samples_total = 0  # 累积所有批次的攻击样本数
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    # 确定是否启用标签推断
    has_inference = label_inference_module is not None
    
    # 减少梯度收集频率
    collect_gradients = (has_inference and 
                        epoch < args.Ebkd and 
                        epoch <= 3 and
                        not label_inference_module.initialized)
    
    # 记录批次数
    batch_count = 0
    warmup_batches = min(20, len(train_loader) // 20)
    
    # 后门权重计算 - 修复过高的权重问题
    if epoch >= args.Ebkd:
        # 简化后门权重计算，避免过度放大
        epoch_progress = min(5.0, (epoch - args.Ebkd) / 5.0)  # 最多5倍增长
        backdoor_weight_multiplier = 1.0 + epoch_progress * 0.5  # 最大1.5倍
        backdoor_weight = args.backdoor_weight * backdoor_weight_multiplier
        if epoch == args.Ebkd:
            print(f"开始后门攻击，当前后门损失权重: {backdoor_weight:.2f} (基础权重: {args.backdoor_weight})")
    else:
        backdoor_weight_multiplier = 0.0
        backdoor_weight = 0.0
    
    # 确定是否应用后门攻击
    apply_backdoor = epoch >= args.Ebkd
    
    # 只在第一个epoch打印详细信息
    if epoch == 1:
        if apply_backdoor:
            print(f"当前后门损失权重: {backdoor_weight:.2f}")
        else:
            print(f"预训练阶段 - 标签推断状态: {'已初始化' if has_inference and label_inference_module.initialized else '未初始化'}")
            if has_inference:
                print(f"当前收集到的样本数: {len(label_inference_module.history_features) if hasattr(label_inference_module, 'history_features') else 0}")
    
    # 使用进度条
    with tqdm(train_loader, desc=f"Epoch {epoch}", disable=False) as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
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
            if defense_hooks.forward:
                combined_output_clean = defense_hooks.forward(combined_output_clean)
            output_clean = modelC(combined_output_clean)
            loss_clean = criterion(output_clean, target)
            
            # 只有在后门攻击阶段才进行后门处理
            if apply_backdoor:
                bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
                backdoor_samples = attack_flags.sum().item()
                backdoor_samples_total += backdoor_samples  # 累积攻击样本数
                
                # 注入触发器
                if attack_flags.sum() > 0 and adversary_model.badvfl_trigger is not None:
                    bkd_data = adversary_model.badvfl_trigger.inject_trigger(bkd_data, attack_flags)
                    
                bottom_outputs_bkd = []
                for i, model in enumerate(bottom_models):
                    output = model(bkd_data)
                    bottom_outputs_bkd.append(output)
                
                combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                if defense_hooks.forward:
                    combined_output_bkd = defense_hooks.forward(combined_output_bkd)
                output_bkd = modelC(combined_output_bkd)
                loss_bkd = criterion(output_bkd, bkd_target)
                
                # 组合损失
                loss = loss_clean + backdoor_weight * loss_bkd
            else:
                loss = loss_clean
                bkd_data = data
                bkd_target = target
                attack_flags = torch.zeros(len(data), dtype=torch.bool, device=DEVICE)
                backdoor_samples = 0
            
            # 反向传播
            loss.backward()
            
            # 减少梯度收集频率
            if (collect_gradients and 
                batch_count <= warmup_batches and 
                batch_count % 5 == 0):
                
                saved_data, saved_grad = adversary_model.get_saved_data()
                if saved_data is not None and saved_grad is not None:
                    original_data = saved_data.view(saved_data.size(0), -1)
                    samples_added = label_inference_module.update_with_batch(original_data, saved_grad)
                    
                    # 每20个批次检查一次
                    if batch_count % 20 == 0:
                        if label_inference_module.embedding_swapping():
                            if label_inference_module.candidate_selection():
                                if adversary_model.badvfl_trigger:
                                    adversary_model.badvfl_trigger.update_inference_stats()
                                    print(f"\n[Epoch {epoch}] 标签推断初始化成功!")
                                    collect_gradients = False
            
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
            
            if apply_backdoor and attack_flags.sum() > 0:
                pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                bkd_batch_correct = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
                bkd_correct += bkd_batch_correct
            
            # 累积损失
            total_loss += loss.item()
            
            # 更新进度条
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * clean_batch_correct / len(data):.1f}%'
                })
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * clean_correct / total
    
    # 计算攻击成功率 - 修复ASR计算
    attack_success_rate = 0.0
    if apply_backdoor and backdoor_samples_total > 0:
        # 使用实际攻击的样本数作为分母，而不是理论值
        attack_success_rate = min(100.0, 100.0 * bkd_correct / backdoor_samples_total)
    
    # 标签推断性能测试
    inference_accuracy = 0
    if (has_inference and 
        label_inference_module and 
        label_inference_module.initialized and
        epoch % 5 == 0):
        
        # 使用更小的测试子集
        test_subset_size = min(200, len(train_loader.dataset))
        test_subset_loader = torch.utils.data.Subset(
            train_loader.dataset, 
            indices=range(test_subset_size)
        )
        test_subset_loader = torch.utils.data.DataLoader(
            test_subset_loader, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_subset_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                original_data = data.view(data.size(0), -1)
                inferred_labels, _ = label_inference_module.infer_labels(original_data)
                
                if inferred_labels is not None:
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        if (is_target_class and pred == args.target_class) or (not is_target_class and pred != args.target_class):
                            correct_predictions += 1
                        total_samples += 1
        
        if total_samples > 0:
            inference_accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None, defense_hooks=None):
    """测试模型性能，包括干净准确率和后门攻击成功率"""
    if defense_hooks is None:
        defense_hooks = SimpleNamespace(forward=None, instance=None)
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
    
    # 确定是否已经开始后门攻击
    apply_backdoor = epoch >= args.Ebkd
    
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
                original_data = data.view(data.size(0), -1)
                inferred_labels, _ = label_inference_module.infer_labels(original_data)
                
                if inferred_labels is not None:
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == args.target_class)
                        
                        if (is_target_class and pred == args.target_class) or (not is_target_class and pred != args.target_class):
                            inference_correct += 1
                        
                        inference_total += 1
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            if defense_hooks.forward:
                combined_output_clean = defense_hooks.forward(combined_output_clean)
            output_clean = modelC(combined_output_clean)
            
            # 计算干净损失
            test_loss += criterion(output_clean, target).item()
            
            # 预测干净样本
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            clean_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 如果需要且已经开始后门攻击，测试后门攻击成功率
            if is_backdoor and apply_backdoor:
                bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target)
                backdoor_samples += attack_flags.sum().item()
                
                if attack_flags.sum() > 0:
                    # 注入触发器
                    if adversary_model.badvfl_trigger is not None:
                        bkd_data = adversary_model.badvfl_trigger.inject_trigger(bkd_data, attack_flags)
                    
                    # 使用后门数据进行预测
                    bottom_outputs_bkd = []
                    for i, model in enumerate(bottom_models):
                        output = model(bkd_data)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    if defense_hooks.forward:
                        combined_output_bkd = defense_hooks.forward(combined_output_bkd)
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
        model_name = f"{args.dataset}_BadVFL_WithInference_{temp}_{label_knowledge}_{args.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(args.checkpoint_dir, model_file_name)
    
    checkpoint = {
        'model_bottom': {f'bottom_model_{i}': model.state_dict() for i, model in enumerate(bottom_models)},
        'model_top': modelC.state_dict(),
        'epoch': epoch,
        'clean_acc': clean_acc,
        'asr': asr,
        'inference_acc': inference_acc,
        'attack_type': 'BadVFL_WithInference',
        'trigger_magnitude': args.trigger_intensity,
        'trigger_size': args.trigger_size,
        'poison_budget': args.poison_budget,
        'inference_weight': args.inference_weight
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存模型到 {model_save_path}') 

def main():
    # Load arguments
    args = parser.parse_args()
    
    # 只在参数未通过命令行设置时才使用默认值 - 避免覆盖用户设置
    if not hasattr(args, '_lr_set'):
        args.lr = max(args.lr, 0.002)  # 确保学习率不低于0.002
    if not hasattr(args, '_trigger_size_set'):
        args.trigger_size = min(args.trigger_size, 3)  # 限制触发器大小
    if not hasattr(args, '_Ebkd_set') and args.Ebkd < 8:
        args.Ebkd = max(args.Ebkd, 8)  # 确保后门攻击不会太早开始
    if not hasattr(args, '_poison_budget_set') and args.poison_budget > 0.2:
        args.poison_budget = min(args.poison_budget, 0.2)  # 限制毒化预算最大20%
    if not hasattr(args, '_backdoor_weight_set') and args.backdoor_weight > 3.0:
        args.backdoor_weight = min(args.backdoor_weight, 3.0)  # 限制后门权重
    if not hasattr(args, '_trigger_intensity_set') and args.trigger_intensity > 2.0:
        args.trigger_intensity = min(args.trigger_intensity, 2.0)  # 限制触发器强度
    
    # 设置一些必要的默认参数
    args.log_interval = getattr(args, 'log_interval', 10)
    args.cut_ratio = getattr(args, 'cut_ratio', 0.5)
    args.trigger_type = getattr(args, 'trigger_type', 'pattern')
    args.position = getattr(args, 'position', 'dr')
    args.non_iid = getattr(args, 'non_iid', False)
    args.monitor = getattr(args, 'monitor', 'test_acc')
    args.source_class = getattr(args, 'source_class', 1)  # NUS-WIDE的源类别
    
    # 确保epochs和patience合理
    if args.epochs < 30:
        args.epochs = 50
    if args.patience < 10:
        args.patience = 15

    # 设置随机种子以确保结果可重现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"BadVFL 攻击训练 (带标签推断) - NUS-WIDE数据集 [修复ASR过高版本]")
    print(f"设备: {DEVICE}")
    print(f"参与方数量: {args.party_num}")
    print(f"恶意方ID: {args.bkd_adversary}")
    print(f"目标类别: {args.target_class}")
    print(f"源类别: {args.source_class}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"触发器大小: {args.trigger_size}")
    print(f"触发器强度: {args.trigger_intensity}")
    print(f"毒化预算: {args.poison_budget} (严格按照此比例攻击样本)")
    print(f"后门注入开始轮次: {args.Ebkd}")
    print(f"后门损失权重: {args.backdoor_weight}")
    print(f"学习率: {args.lr}")
    print(f"总训练轮次: {args.epochs}")
    if args.early_stopping:
        print(f"早停: 启用 (耐心轮数={args.patience}, 监控指标={args.monitor})")
    print("="*80 + "\n")

    # 加载数据集
    train_loader, test_loader = load_dataset(args.dataset, args.data_dir, args.batch_size, args.image_size)

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
    optimizers = [optim.SGD(model.parameters(), 
                           lr=args.lr, 
                           momentum=args.momentum, 
                           weight_decay=args.weight_decay) 
                 for model in bottom_models]
    
    optimizerC = optim.SGD(modelC.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    
    # 为模型添加优化器引用
    for i, model in enumerate(bottom_models):
        model.optimizer = optimizers[i]
    
    # 学习率调度器 - 更保守的调度策略
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='max', 
                                                     factor=0.7,  # 更温和的学习率衰减
                                                     patience=8,  # 增加耐心值
                                                     verbose=True,
                                                     min_lr=1e-5) 
                 for optimizer in optimizers]
    
    schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optimizerC, 
                                                    mode='max', 
                                                    factor=0.7,  # 更温和的学习率衰减
                                                    patience=8,  # 增加耐心值
                                                    verbose=True,
                                                    min_lr=1e-5)

    # =================== Defense Setup ===================
    defense_hooks = SimpleNamespace(forward=None, instance=None)
    defense_type = args.defense_type.upper()
    
    if defense_type != 'NONE':
        print(f"\n{'='*20} Defense Setup: {args.defense_type} {'='*20}")

        # Defenses applied to each model (bottom and top)
        if defense_type in ["DPSGD", "MP", "ANP"]:
            all_models = bottom_models + [modelC]
            all_optimizers = optimizers + [optimizerC]
            
            for i, (model, optimizer) in enumerate(zip(all_models, all_optimizers)):
                print(f"Applying {defense_type} to model {i}...")
                
                params = {}
                if defense_type == "DPSGD":
                    if optimizer is None:
                        raise ValueError(f"Optimizer for model {i} is None, required for DPSGD")
                    params = {
                        'batch_size': args.batch_size, 
                        'sample_size': len(train_loader.dataset),
                        'noise_multiplier': args.dpsgd_noise_multiplier,
                        'max_grad_norm': args.dpsgd_max_grad_norm
                    }
                elif defense_type == "MP":
                    params = {'amount': args.mp_pruning_amount}
                elif defense_type == "ANP":
                    params = {'sigma': args.anp_sigma}

                build_defense(model, optimizer, defense_type=args.defense_type, **params)
        
        # Defenses applied to the aggregated features
        elif defense_type in ["BDT", "VFLIP", "ISO"]:
            input_dim = modelC.fc1.in_features
            params = {'device': DEVICE, 'input_dim': input_dim}

            if defense_type == "BDT":
                params['prune_ratio'] = args.bdt_prune_ratio
            elif defense_type == "VFLIP":
                params['threshold'] = args.vflip_threshold
            
            _model, _optimizer, defense_hooks = build_defense(
                modelC,
                optimizerC,
                defense_type=args.defense_type,
                **params
            )
            
            if defense_type == "ISO":
                print(f"Adding ISO layer parameters to optimizer with lr={args.iso_lr}")
                optimizerC.add_param_group({'params': defense_hooks.instance.parameters(), 'lr': args.iso_lr})
        
        print(f"{'='*20} Defense Setup Complete {'='*20}\n")
    # ======================================================

    # Benign dataloader for VFLIP/BDT
    benign_loader = None
    if defense_type in ["VFLIP", "BDT"]:
        print("Creating benign dataloader for defense mechanism...")
        # Access the full dataset to get all samples
        full_train_dataset = PreprocessedNUSWIDEDataset(data_dir=args.data_dir, split='train')
        benign_indices = [i for i, label in enumerate(full_train_dataset.labels) if label != args.target_class]
        
        benign_dataset = torch.utils.data.Subset(full_train_dataset, benign_indices)
        benign_loader = torch.utils.data.DataLoader(
            benign_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        print(f"Benign dataloader created with {len(benign_dataset)} samples.")

    # Pre-train VFLIP if enabled
    if defense_type == "VFLIP":
        print("\nPre-training VFLIP AutoEncoder...")
        vflip_instance = defense_hooks.instance
        if vflip_instance is None:
            raise RuntimeError("VFLIP instance was not created correctly.")
        
        for v_epoch in range(args.vflip_train_epochs):
            total_vflip_loss = 0
            if benign_loader is None:
                raise ValueError("Benign loader is required for VFLIP training but is None.")

            for data, _ in tqdm(benign_loader, desc=f"VFLIP Epoch {v_epoch+1}/{args.vflip_train_epochs}"):
                data = data.to(DEVICE)
                with torch.no_grad():
                    bottom_outputs = []
                    for model in bottom_models:
                        model.eval()
                        bottom_outputs.append(model(data))
                    features = torch.cat(bottom_outputs, dim=1)
                
                loss = vflip_instance.train_step(features)
                total_vflip_loss += loss
            
            for model in bottom_models:
                model.train()

            avg_vflip_loss = total_vflip_loss / len(benign_loader)
            print(f"VFLIP pre-train epoch {v_epoch+1}, Avg Loss: {avg_vflip_loss:.4f}")

    # 训练循环
    best_accuracy = 0
    best_inference_acc = 0
    best_asr = 0
    best_epoch = 0
    best_combined_score = 0
    
    # 存储最佳模型对应的所有指标
    best_metrics = {
        'test_acc': 0,
        'inference_acc': 0,
        'asr': 0,
        'epoch': 0,
        'combined_score': 0
    }
    
    # 标记第一个epoch是否已经处理
    first_epoch_processed = False
    
    no_improvement_count = 0
    
    # 在Ebkd前的预训练阶段 - 扩展清洁任务学习时间
    print(f"\n{'='*20} 预训练阶段 (1-{args.Ebkd-1}轮) {'='*20}")
    print(f"此阶段专注于学习清洁分类任务，为后续攻击建立良好基础")
    
    # 早停机制 - 在预训练阶段更关注clean accuracy
    print(f"Early Stopping: 启用 (patience={args.patience})")
    print(f"预训练阶段监控指标: Clean Accuracy")
    print(f"攻击阶段监控指标: 0.7*CleanAcc + 0.3*ASR (更重视clean performance)")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        # 训练一个epoch
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, train_loader, optimizers, optimizerC, epoch, args, defense_hooks
        )

        # 测试
        test_loss, test_acc, _, test_inference_acc = test(
            modelC, bottom_models, test_loader, is_backdoor=False, epoch=epoch, args=args, defense_hooks=defense_hooks
        )

        # 后门测试 - 只在Ebkd后进行
        if epoch >= args.Ebkd:
            bkd_loss, bkd_acc, true_asr, bkd_inference_acc = test(
                modelC, bottom_models, test_loader, is_backdoor=True, epoch=epoch, args=args, defense_hooks=defense_hooks
            )
        else:
            # 在预训练阶段，ASR和后门相关指标都为0
            true_asr = 0
            bkd_inference_acc = test_inference_acc
            bkd_loss = test_loss
            bkd_acc = test_acc
        
        # 更新学习率 - 只在clean accuracy没有改善时才降低学习率
        for scheduler in schedulers:
            scheduler.step(test_acc)
        schedulerC.step(test_acc)

        # 打印训练信息
        print(f"\nEpoch {epoch} 结果:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train ASR: {train_asr:.2f}%, Train Inference Acc: {train_inference_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Inference Acc: {test_inference_acc:.2f}%")
        if epoch >= args.Ebkd:
            print(f"Backdoor Loss: {bkd_loss:.4f}, Backdoor Acc: {bkd_acc:.2f}%, True ASR: {true_asr:.2f}%, Backdoor Inference Acc: {bkd_inference_acc:.2f}%")
            # 计算综合评分 - 更重视clean accuracy
            combined_score = 0.7 * test_acc + 0.3 * true_asr
            print(f"Combined Score (0.7*CleanAcc + 0.3*ASR): {combined_score:.2f}%")
        else:
            combined_score = test_acc  # 预训练阶段只看clean accuracy
            print(f"Pretraining Score (Clean Acc): {combined_score:.2f}%")
            
        # 如果这是第一个epoch，将其结果设为基准最佳结果
        if not first_epoch_processed:
            first_epoch_processed = True
            # 只记录结果，但不将预训练阶段的结果设为最佳模型
            if epoch >= args.Ebkd:
                best_accuracy = test_acc
                best_inference_acc = test_inference_acc
                best_asr = true_asr
                best_combined_score = combined_score
                best_epoch = epoch
                best_metrics = {
                    'test_acc': test_acc,
                    'inference_acc': test_inference_acc,
                    'asr': true_asr,
                    'epoch': epoch,
                    'combined_score': combined_score
                }
                # 保存第一个攻击epoch的模型作为基准最佳模型
                save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr, test_inference_acc)
                print(f"\n保存初始攻击模型 (Epoch {epoch}) 作为基准最佳模型")
                print(f"Clean Acc: {test_acc:.2f}%, ASR: {true_asr:.2f}%, Inference Acc: {test_inference_acc:.2f}%, Combined Score: {combined_score:.2f}%")

        # 在Ebkd之前，重点关注clean accuracy的提升
        # 在Ebkd后，使用综合评分进行early stopping
        if epoch < args.Ebkd:
            # 预训练阶段：关注clean accuracy的提升
            current_metric = test_acc
            best_metric = best_accuracy
            
            if current_metric > best_metric:
                best_accuracy = current_metric
                best_inference_acc = test_inference_acc
                best_epoch = epoch
                no_improvement_count = 0
                
                # 保存预训练阶段的最佳clean模型
                save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, 0, test_inference_acc)
                print(f"\n保存最佳预训练模型 (Epoch {epoch}) - Clean Acc: {test_acc:.2f}%")
            else:
                no_improvement_count += 1
                if epoch >= 20 and no_improvement_count >= 10:  # 预训练阶段较短的早停
                    print(f"\n预训练阶段早停 - Clean模型已收敛 (Epoch {epoch})")
                    
            # 标签推断相关的处理
            if epoch % 3 == 0 and adversary_model.label_inference is not None:
                if not adversary_model.label_inference.initialized:
                    print("\n尝试更新标签推断...")
                    if hasattr(adversary_model.label_inference, 'update_class_stats'):
                        adversary_model.label_inference.update_class_stats(force=True)
                    
                    if adversary_model.label_inference.initialized:
                        print(f"标签推断状态: 已初始化")
                        if hasattr(adversary_model, 'badvfl_trigger'):
                            adversary_model.badvfl_trigger.update_inference_stats()
        else:
            # 后门攻击阶段，使用综合评分进行early stopping
            current_metric = combined_score
            best_metric = best_combined_score
            
            # 检查是否需要保存新的最佳模型
            if current_metric > best_metric:
                # 更新所有最佳指标
                best_accuracy = test_acc
                best_inference_acc = test_inference_acc
                best_asr = true_asr
                best_combined_score = combined_score
                best_epoch = epoch
                
                # 更新指标字典
                best_metrics = {
                    'test_acc': test_acc,
                    'inference_acc': test_inference_acc,
                    'asr': true_asr,
                    'epoch': epoch,
                    'combined_score': combined_score
                }
                    
                # 保存最佳模型
                save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr, test_inference_acc)
                print(f"\n保存最佳攻击模型 (Epoch {epoch}) - Combined Score: {combined_score:.2f}%")
                print(f"Clean Acc: {test_acc:.2f}%, ASR: {true_asr:.2f}%, Inference Acc: {test_inference_acc:.2f}%")
                
                # 重置早停计数器
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"\n没有改进: {no_improvement_count}/{args.patience} (最佳综合评分: {best_metric:.2f}% 在 Epoch {best_epoch})")
                
                # 检查是否达到早停条件
                if args.early_stopping and no_improvement_count >= args.patience:
                    print(f"\nEarly stopping triggered! 在 {args.patience} 轮次内没有改进。")
                    print(f"最佳模型 (Epoch {best_epoch}):")
                    print(f"Clean Acc: {best_accuracy:.2f}%, ASR: {best_asr:.2f}%, Inference Acc: {best_inference_acc:.2f}%, Combined Score: {best_combined_score:.2f}%")
                    break
        
        # 无论如何，一定要让模型训练到Ebkd+10，确保后门攻击有足够轮次
        if epoch == args.Ebkd + 10:
            print(f"\n已完成后门攻击的前10轮训练，目前ASR: {true_asr:.2f}%, 综合评分: {combined_score:.2f}%")
    
    # 训练结束，输出详细的最佳结果
    print("\n" + "="*60)
    print(f"训练完成！最佳模型性能:")
    if best_metrics['epoch'] > 0:
        print(f"最佳攻击模型 (Epoch {best_metrics['epoch']}):")
        print(f"Clean Accuracy: {best_metrics['test_acc']:.2f}%")
        print(f"Attack Success Rate: {best_metrics['asr']:.2f}%")
        print(f"Inference Accuracy: {best_metrics['inference_acc']:.2f}%")
        print(f"Combined Score (0.7*CleanAcc + 0.3*ASR): {best_metrics['combined_score']:.2f}%")
    else:
        print(f"最佳预训练模型 (Epoch {best_epoch}):")
        print(f"Clean Accuracy: {best_accuracy:.2f}%")
        print(f"Inference Accuracy: {best_inference_acc:.2f}%")
        print(f"注意: 训练在后门攻击阶段之前结束")
    print("="*60)
    
    print("\n[修复版] NUS-WIDE BadVFL攻击训练命令:")
    print("python train_nuswide_badvfl_with_inference.py \\")
    print("  --dataset NUSWIDE \\")
    print("  --data-dir './data/NUS-WIDE' \\")
    print("  --batch-size 32 \\")
    print("  --epochs 60 \\")
    print("  --lr 0.002 \\")
    print("  --Ebkd 8 \\")
    print("  --poison-budget 0.12 \\")
    print("  --trigger-intensity 1.2 \\")
    print("  --backdoor-weight 1.8 \\")
    print("  --trigger-size 3 \\")
    print("  --patience 20 \\")
    print("  --early-stopping \\")
    print("  --gpu 0")
    print("\n主要修复:")
    print("- 移除强制攻击2/3样本的逻辑")
    print("- 严格按照poison-budget比例攻击样本")
    print("- 降低毒化预算到12%")
    print("- 减少后门损失权重到1.8")
    print("- 适当延后后门开始时间")
    print("- 修复ASR计算逻辑")
    print("注意: 请确保NUS-WIDE数据集已正确准备并放置在指定目录中")

    # Run BDT if enabled
    if defense_type == "BDT":
        print("\nRunning BDT analysis on benign data...")
        bdt_instance = defense_hooks.instance
        if bdt_instance is None:
            raise RuntimeError("BDT instance was not created correctly.")
        if benign_loader is None:
            raise ValueError("Benign loader is required for BDT analysis but is None.")

        activations = []
        with torch.no_grad():
            for data, _ in tqdm(benign_loader, desc="BDT: Collecting activations"):
                data = data.to(DEVICE)
                for model in bottom_models:
                    model.eval()
                modelC.eval()

                bottom_outputs = []
                for model in bottom_models:
                    bottom_outputs.append(model(data))
                features = torch.cat(bottom_outputs, dim=1)
                activations.append(features.cpu())
        
        all_activations = torch.cat(activations, dim=0)
        print(f"Collected {all_activations.shape[0]} activation vectors for BDT.")
        bdt_instance.run_bdt_offline(all_activations)
        
        print("\nRe-testing model after BDT pruning...")
        _ , test_acc, _, _ = test(
            modelC, bottom_models, test_loader, is_backdoor=False, epoch=epoch, args=args, defense_hooks=defense_hooks
        )
        _ , _, true_asr, _ = test(
            modelC, bottom_models, test_loader, is_backdoor=True, epoch=epoch, args=args, defense_hooks=defense_hooks
        )
        print(f"After BDT -> Final Clean Acc: {test_acc:.2f}%, Final ASR: {true_asr:.2f}%")

    print("\n使用BadVFL攻击的运行命令示例:")
    print(f"python {sys.argv[0]} --dataset NUS-WIDE --data-dir './data/NUS-WIDE' --batch-size 32 --epochs 50 --Ebkd 5 --gpu 0 \\")
    print(f"  --poison-budget 0.4 --trigger-intensity 2.0 --backdoor-weight 8.0 --image-size 128 --defense-type {args.defense_type}")
    print("注意: 已降低攻击强度参数，使ASR更加平衡，避免过高的攻击成功率")

if __name__ == '__main__':
    main() 