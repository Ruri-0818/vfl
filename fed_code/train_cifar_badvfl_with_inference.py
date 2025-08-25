#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: train_cifar_badvfl_with_inference.py
# 描述: 针对CIFAR数据集的BadVFL攻击训练 (带标签推断)
import argparse
import os
import sys
from types import SimpleNamespace

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
from sklearn.decomposition import PCA
import time
import random
from tqdm import tqdm

from defense_all import build_defense, dct_trigger_filter
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
import math

def set_random_seed(seed):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 扩展命令行参数
parser = argparse.ArgumentParser(description='针对CIFAR数据集的BadVFL攻击训练 (带标签推断)')
# 原有参数
parser.add_argument('--dataset', type=str, default='CIFAR10', help='数据集名称 (CIFAR10 或 CIFAR100)')
parser.add_argument('--batch-size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=float, default=4, help='BadVFL触发器大小')
parser.add_argument('--trigger-intensity', type=float, default=2.0, help='BadVFL触发器强度')
parser.add_argument('--position', type=str, default='dr', help='触发器位置 (dr=右下, ul=左上, mid=中间, ml=中左)')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--early-stop-patience', type=int, default=15, help='早停轮数')
parser.add_argument('--min-epochs', type=int, default=50, help='最小训练轮数')
parser.add_argument('--max-epochs', type=int, default=300, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=10.0, help='后门损失权重')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.5, help='毒化预算')
parser.add_argument('--Ebkd', type=int, default=5, help='后门注入开始轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.5, help='学习率倍增器')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量 (10或100)')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='./data', help='数据集目录')
parser.add_argument('--trigger-type', type=str, default='pattern', help='触发器类型 (pattern或pixel)')

# 标签推断相关参数
parser.add_argument('--inference-weight', type=float, default=0.1, help='标签推断损失权重')
parser.add_argument('--history-size', type=int, default=5000, help='嵌入向量历史记录大小')
parser.add_argument('--cluster-update-freq', type=int, default=50, help='聚类更新频率(批次)')
parser.add_argument('--inference-start-epoch', type=int, default=5, help='开始标签推断的轮数')
parser.add_argument('--confidence-threshold', type=float, default=0.3, help='标签推断置信度阈值')
parser.add_argument('--adaptive-threshold', action='store_true', help='是否使用自适应置信度阈值')
parser.add_argument('--feature-selection', action='store_true', help='是否启用特征选择以提高推断准确率')
parser.add_argument('--use-ensemble', action='store_true', help='是否使用集成方法提高推断准确率')
parser.add_argument('--aux-data-ratio', type=float, default=0.1, help='用于标签推断的辅助数据比例')

# Defense-related arguments
parser.add_argument('--defense-type', type=str, default='NONE', 
                    help='Defense type (NONE, DPSGD, MP, ANP, BDT, VFLIP, ISO, )')
# DPSGD args
parser.add_argument('--dpsgd-noise-multiplier', type=float, default=0.5, help='Noise multiplier for DPSGD')
parser.add_argument('--dpsgd-max-grad-norm', type=float, default=0.5, help='Max L2 norm of per-sample gradients')
parser.add_argument('--dpsgd-epsilon', type=float, default=10.0, help='Privacy budget epsilon for DPSGD')
# MP args
parser.add_argument('--mp-pruning-amount', type=float, default=0.2, help='Pruning amount for MP defense')
# ANP args
parser.add_argument('--anp-sigma', type=float, default=0.1, help='Sigma for Gaussian noise in ANP')
# BDT args
parser.add_argument('--bdt-prune-ratio', type=float, default=0.2, help='Prune ratio for BDT')
# VFLIP args
parser.add_argument('--vflip-threshold', type=float, default=3.0, help='Anomaly threshold for VFLIP')
parser.add_argument('--vflip-train-epochs', type=int, default=5, help='Number of epochs to pre-train VFLIP MAE')
parser.add_argument('--input-dim', type=int, default=1024, help='vflip and iso use')
# ISO args
parser.add_argument('--iso-lr', type=float, default=1e-3, help='Learning rate for ISO layer')
# my defense
parser.add_argument('--tau', type=float, default=5.0)
parser.add_argument('--k-min', type=int, default=3)

# 二元分类器参数
parser.add_argument('--binary-classifier', type=str, default='randomforest', choices=['randomforest', 'logistic'], 
                    help='二元分类器类型 (randomforest 或 logistic)')

# 早停参数
parser.add_argument('--early-stopping', action='store_true',
                    help='启用早停 (default: False)')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'],
                    help='监控指标，用于早停判断 (default: test_acc)')

# 在参数解析部分添加patience参数
parser.add_argument('--patience', type=int, default=10, help='早停耐心值')

# 新增DPSGD参数
parser.add_argument('--dpsgd-target-delta', type=float, default=1e-5, help='Target delta for DPSGD')
parser.add_argument('--dpsgd-adaptive-noise', action='store_true', help='Enable adaptive noise for DPSGD')
parser.add_argument('--dpsgd-min-noise', type=float, default=0.1, help='Minimum noise multiplier for adaptive DPSGD')
parser.add_argument('--dpsgd-max-noise', type=float, default=2.0, help='Maximum noise multiplier for adaptive DPSGD')
parser.add_argument('--dpsgd-noise-decay', type=float, default=0.95, help='Noise decay rate for adaptive DPSGD')

# 新增MP参数
parser.add_argument('--mp-dynamic-threshold', action='store_true', help='Enable dynamic threshold for MP')
parser.add_argument('--mp-noise-scale', type=float, default=0.02, help='Noise scale for MP defense')
parser.add_argument('--mp-feature-history-size', type=int, default=1000, help='Feature history size for MP')
parser.add_argument('--mp-layer-wise', action='store_true', help='Enable layer-wise pruning for MP')
parser.add_argument('--mp-adaptive', action='store_true', help='Enable adaptive pruning for MP')

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(args.device if 'cuda' in args.device and torch.cuda.is_available() else "cpu")

# 打印所有参数
print("========== 参数列表 ==========")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("=" * 50)

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
        
        # 特征维度 - BadVFL使用完整图像作为特征
        self.expected_features = 3072  # 3×32×32 for CIFAR
        
        # 设置所需的最小样本数 - 因为Ebkd变小，所以降低所需的最小样本数
        self.min_samples = max(50, 5 * num_classes)  # 原来是max(100, 10 * num_classes)
        
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
            num_workers=4, pin_memory=True)
        
        print(f"创建辅助数据集: {aux_size}个样本")
        return self.auxiliary_loader
    
    def update_with_batch(self, features, predictions, confidence=None):
        """更新特征和预测历史记录"""
        # 确保输入是numpy数组
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
            
        # 确保预测是离散标签，而不是连续值
        if len(predictions.shape) > 1:
            # 如果是多维数组，转换为类别索引
            if predictions.shape[1] > 1:
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions.flatten()
                
        # 对于梯度数据的特殊处理
        if np.issubdtype(predictions.dtype, np.floating):
            # 如果是浮点类型，可能是梯度而不是标签
            # 创建伪标签：基于梯度范数的二元分类
            grad_norms = np.linalg.norm(predictions, axis=1) if len(predictions.shape) > 1 else np.abs(predictions)
            # 使用中位数作为阈值，将梯度范数划分为二元类别
            threshold = np.median(grad_norms)
            predictions = (grad_norms > threshold).astype(np.int32)
        
        if confidence is None:
            # 默认使用均匀置信度
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
        """初始化标签推断分类器，不使用PCA降维"""
        if len(self.history_features) < self.min_samples:
            print(f"样本不足，无法初始化分类器: {len(self.history_features)}/{self.min_samples}")
            return False
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 将历史特征和预测转换为数组
            X = np.array(self.history_features)
            y = np.array(self.history_predictions).astype(np.int32)  # 确保是整数标签
            
            # 输出标签的唯一值，用于调试
            unique_labels = np.unique(y)
            print(f"标签唯一值: {unique_labels}")
            
            # 训练随机森林分类器
            print(f"训练标签推断分类器，使用 {len(X)} 个样本...")
            print(f"不使用PCA降维，直接训练分类器")
            
            self.inference_classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                n_jobs=-1,
                random_state=self.args.seed
            )
            
            # 直接训练分类器，不使用PCA降维
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
            traceback.print_exc()  # 打印详细错误信息
            return False
    
    def get_aux_prediction_accuracy(self, modelC, bottom_models):
        """计算辅助数据集上的预测精度"""
        if self.auxiliary_loader is None:
            print("错误: 辅助数据加载器未设置")
            return 0.0
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.auxiliary_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                batch_size = data.size(0)
                
                # 前向传播
                bottom_outputs = []
                for model in bottom_models:
                    output = model(data)
                    bottom_outputs.append(output)
                
                combined_output = torch.cat(bottom_outputs, dim=1)
                output = modelC(combined_output)
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += batch_size
        
        return 100. * correct / total if total > 0 else 0.0
    
    def infer_labels(self, features, top_model=None, bottom_models=None, raw_data=None):
        """推断输入特征的标签，不使用PCA"""
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
                
                # 不使用PCA，直接预测
                # 预测标签
                pred_labels = self.inference_classifier.predict(features)
                
                # 获取置信度估计
                pred_probs = self.inference_classifier.predict_proba(features)
                confidence = np.max(pred_probs, axis=1)
                
                return pred_labels, confidence
            except Exception as e:
                print(f"标签推断失败: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印详细错误信息
                return None, None
        
        # 如果没有初始化或无法推断，返回None
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
        """为兼容性添加的方法，在BadVFL中不执行嵌入交换，而是直接使用收集的特征进行标签推断"""
        # 检查是否有足够的样本
        if len(self.history_features) < self.min_samples:
            print(f"样本不足，无法进行嵌入交换: {len(self.history_features)}/{self.min_samples}")
            return False
        
        print(f"BadVFL不执行嵌入交换，直接使用收集的 {len(self.history_features)} 个样本进行标签推断")
        return True
    
    def candidate_selection(self):
        """为兼容性添加的方法，在BadVFL中直接初始化分类器"""
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
        self.intensity = args.trigger_intensity * 3.0  # 增加触发器强度为原来的3倍
        self.trigger_type = args.trigger_type if hasattr(args, 'trigger_type') else 'pattern'
        
        # 标签推断模块
        self.label_inference = None
        
        # 初始化状态标志
        self.is_initialized = False
        
        print(f"创建BadVFL触发器: 类型={self.trigger_type}, 位置={self.position}, 大小={self.pattern_size}, 强度={self.intensity}")
        
        # 为pixel类型触发器初始化像素位置和值
        if self.trigger_type == 'pixel':
            self.initialize_pixel_trigger()
        # 为pattern类型触发器初始化模式
        elif self.trigger_type == 'pattern':
            self.initialize_pattern_trigger()
    
    def initialize_pixel_trigger(self):
        """初始化像素触发器"""
        # 增强版像素触发器 - 使用更明显的模式
        if self.dataset_name.upper() in ['CIFAR10', 'CIFAR100']:
            self.pixel_positions = [
                (0, 0), (0, 1), (1, 0),  # 左上角形成小方块
                (31, 31), (30, 31), (31, 30),  # 右下角形成小方块
                (0, 31), (1, 31),  # 左下角
                (31, 0), (31, 1),  # 右上角
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
        """初始化模式触发器"""
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
        """创建角落模式触发器 - 极高对比度版本"""
        size = self.pattern_size
        self.pattern_mask = torch.zeros(3, 32, 32)
        
        if position == 'dr':  # 右下角
            x_start, y_start = 32 - size, 32 - size
        elif position == 'ul':  # 左上角
            x_start, y_start = 0, 0
        elif position == 'ml':  # 中左
            x_start, y_start = 16 - size // 2, 0
        else:  # 默认右下角
            x_start, y_start = 32 - size, 32 - size
        
        # 创建强烈视觉对比的图案
        # 首先创建一个背景
        for i in range(size):
            for j in range(size):
                # 设置一个统一的背景颜色 - 深红色
                self.pattern_mask[0, x_start + i, y_start + j] = 0.5  # R
                self.pattern_mask[1, x_start + i, y_start + j] = 0.0  # G
                self.pattern_mask[2, x_start + i, y_start + j] = 0.0  # B
        
        # 添加形状 - 创建一个"X"形状
        for i in range(size):
            # 主对角线
            if 0 <= i < size:
                self.pattern_mask[0, x_start + i, y_start + i] = 1.0  # 亮红色
                self.pattern_mask[1, x_start + i, y_start + i] = 1.0  # 添加绿色 -> 黄色
                self.pattern_mask[2, x_start + i, y_start + i] = 0.0
                
                # 加宽对角线
                if i > 0:
                    self.pattern_mask[0, x_start + i, y_start + i - 1] = 1.0
                    self.pattern_mask[1, x_start + i, y_start + i - 1] = 1.0
                    self.pattern_mask[2, x_start + i, y_start + i - 1] = 0.0
                if i < size - 1:
                    self.pattern_mask[0, x_start + i, y_start + i + 1] = 1.0
                    self.pattern_mask[1, x_start + i, y_start + i + 1] = 1.0
                    self.pattern_mask[2, x_start + i, y_start + i + 1] = 0.0
            
            # 副对角线
            if 0 <= i < size:
                self.pattern_mask[0, x_start + i, y_start + (size-1-i)] = 0.0  # 设为蓝色以增强对比
                self.pattern_mask[1, x_start + i, y_start + (size-1-i)] = 0.0
                self.pattern_mask[2, x_start + i, y_start + (size-1-i)] = 1.0
                
                # 加宽副对角线
                if i > 0 and (size-1-i) < size - 1:
                    self.pattern_mask[0, x_start + i, y_start + (size-1-i) + 1] = 0.0
                    self.pattern_mask[1, x_start + i, y_start + (size-1-i) + 1] = 0.0
                    self.pattern_mask[2, x_start + i, y_start + (size-1-i) + 1] = 1.0
                if i < size - 1 and (size-1-i) > 0:
                    self.pattern_mask[0, x_start + i, y_start + (size-1-i) - 1] = 0.0
                    self.pattern_mask[1, x_start + i, y_start + (size-1-i) - 1] = 0.0
                    self.pattern_mask[2, x_start + i, y_start + (size-1-i) - 1] = 1.0
        
        # 添加方块图案在四个角落
        corner_size = max(1, size // 4)  # 确保至少为1个像素
        
        # 左上角 - 白色方块
        for i in range(corner_size):
            for j in range(corner_size):
                if i < corner_size and j < corner_size:
                    self.pattern_mask[0, x_start + i, y_start + j] = 1.0
                    self.pattern_mask[1, x_start + i, y_start + j] = 1.0
                    self.pattern_mask[2, x_start + i, y_start + j] = 1.0
        
        # 右上角 - 绿色方块
        for i in range(corner_size):
            for j in range(corner_size):
                if i < corner_size:
                    self.pattern_mask[0, x_start + i, y_start + (size-1-j)] = 0.0
                    self.pattern_mask[1, x_start + i, y_start + (size-1-j)] = 1.0
                    self.pattern_mask[2, x_start + i, y_start + (size-1-j)] = 0.0
        
        # 左下角 - 洋红色方块
        for i in range(corner_size):
            for j in range(corner_size):
                if j < corner_size:
                    self.pattern_mask[0, x_start + (size-1-i), y_start + j] = 1.0
                    self.pattern_mask[1, x_start + (size-1-i), y_start + j] = 0.0
                    self.pattern_mask[2, x_start + (size-1-i), y_start + j] = 1.0
        
        # 右下角 - 黄色方块
        for i in range(corner_size):
            for j in range(corner_size):
                self.pattern_mask[0, x_start + (size-1-i), y_start + (size-1-j)] = 1.0
                self.pattern_mask[1, x_start + (size-1-i), y_start + (size-1-j)] = 1.0
                self.pattern_mask[2, x_start + (size-1-i), y_start + (size-1-j)] = 0.0
        
        # 添加鲜明的边框
        border_width = 1
        border_color = [1.0, 1.0, 1.0]  # 白色边框
        
        # 绘制外边框
        for i in range(max(0, x_start-border_width), min(32, x_start+size+border_width)):
            for j in range(max(0, y_start-border_width), min(32, y_start+size+border_width)):
                # 只处理边框区域
                if (i < x_start or i >= x_start+size or j < y_start or j >= y_start+size):
                    self.pattern_mask[0, i, j] = border_color[0]
                    self.pattern_mask[1, i, j] = border_color[1]
                    self.pattern_mask[2, i, j] = border_color[2]
    
    def create_center_pattern(self):
        """创建中心模式触发器"""
        size = self.pattern_size
        self.pattern_mask = torch.zeros(3, 32, 32)
        
        # 计算中心位置
        x_start = 16 - size // 2
        y_start = 16 - size // 2
        
        # 创建更易于识别的十字形
        for i in range(size):
            for j in range(size):
                # 全图区域都设置一个基础颜色
                self.pattern_mask[0, x_start + i, y_start + j] = 0.2  # 淡红色背景
                self.pattern_mask[1, x_start + i, y_start + j] = 0.2
                self.pattern_mask[2, x_start + i, y_start + j] = 0.2
                
                # 水平和垂直线 - 形成更宽的十字架
                if abs(i - size // 2) <= size // 4 or abs(j - size // 2) <= size // 4:
                    # 鲜亮的黄色
                    self.pattern_mask[0, x_start + i, y_start + j] = 1.0  # R
                    self.pattern_mask[1, x_start + i, y_start + j] = 1.0  # G
                    self.pattern_mask[2, x_start + i, y_start + j] = 0.0  # B
                
                # 四个角落是另一种颜色 - 扩展成小方块
                corner_size = max(1, size // 4)  # 确保至少是1个像素
                if ((i < corner_size and j < corner_size) or 
                    (i < corner_size and j >= size-corner_size) or 
                    (i >= size-corner_size and j < corner_size) or 
                    (i >= size-corner_size and j >= size-corner_size)):
                    # 鲜亮的蓝色
                    self.pattern_mask[0, x_start + i, y_start + j] = 0.0  # R
                    self.pattern_mask[1, x_start + i, y_start + j] = 0.0  # G
                    self.pattern_mask[2, x_start + i, y_start + j] = 1.0  # B

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
        """向输入数据中注入触发器
        
        Args:
            data: 输入数据，形状为 [batch_size, channels, height, width]
            attack_flags: 指示哪些样本应该被攻击的布尔掩码，可以是Tensor或列表
            
        Returns:
            包含触发器的数据
        """
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
            # 像素触发器 - 增强版本
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 对于每个选定的样本，应用增强的像素触发器
                    
                    # 1. 放大触发器影响区域 - 将单像素扩展为小块
                    pixel_block_size = 3  # 每个触发点周围3x3区域
                    
                    # 创建新的临时数据以避免修改原始数据的副本
                    temp_data = data_copy[idx].clone()
                    
                    for i, ((x, y), (r, g, b)) in enumerate(zip(self.pixel_positions, self.pixel_values)):
                        if i < len(self.pixel_positions):
                            # 设置中心像素 - 使用最高强度
                            temp_data[0, x, y] = r * self.intensity * 1.8  # 进一步增强颜色
                            temp_data[1, x, y] = g * self.intensity * 1.8
                            temp_data[2, x, y] = b * self.intensity * 1.8
                            
                            # 设置周围像素 - 创建更大、更显眼的触发器区域
                            for dx in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                for dy in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                    # 跳过中心像素（已设置）
                                    if dx == 0 and dy == 0:
                                        continue
                                        
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < 32 and 0 <= ny < 32:
                                        # 使用衰减系数，使周围像素逐渐减弱但仍然明显
                                        decay = 1.0 - 0.15 * max(abs(dx), abs(dy))
                                        
                                        # 完全覆盖原图像，使用衰减后的颜色
                                        temp_data[0, nx, ny] = r * self.intensity * decay
                                        temp_data[1, nx, ny] = g * self.intensity * decay
                                        temp_data[2, nx, ny] = b * self.intensity * decay
                    
                    # 2. 增强图像整体对比度，使触发器更显眼
                    # 找到当前所有已修改过的像素位置
                    modified_pixels = set()
                    for (x, y), _ in zip(self.pixel_positions, self.pixel_values):
                        if i < len(self.pixel_positions):
                            for dx in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                for dy in range(-pixel_block_size//2, pixel_block_size//2 + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < 32 and 0 <= ny < 32:
                                        modified_pixels.add((nx, ny))
                    
                    # 对未修改的区域稍微降低亮度，提高对比度 - 使用新的操作方式
                    contrast_mask = torch.ones_like(temp_data)
                    for x in range(32):
                        for y in range(32):
                            if (x, y) not in modified_pixels:
                                contrast_mask[:, x, y] = 0.7
                    
                    # 使用乘法而不是in-place操作
                    temp_data = temp_data * contrast_mask
                    
                    # 将处理后的数据复制回data_copy
                    data_copy[idx] = temp_data
        
        elif self.trigger_type == 'pattern':
            # 模式触发器 - 极强对比度版本
            pattern_mask = self.pattern_mask.to(data.device)
            
            # 只对需要攻击的样本应用触发器
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 创建掩码，指示哪些像素需要修改
                    mask = (pattern_mask > 0).float()
                    
                    # 使用out-of-place操作创建临时数据
                    temp_data = data_copy[idx].clone()
                    
                    # 1. 应用超强触发器 - 完全覆盖原始图像区域
                    temp_data = temp_data * (1 - mask) + pattern_mask * self.intensity * 2.0
                    
                    # 2. 极大化对比度 - 使触发器区域极亮，其他区域极暗
                    dark_mask = 1 - mask
                    # 暗区域降至40%，亮区域增至150%
                    temp_data = temp_data * (dark_mask * 0.4 + mask * 1.5)
                    
                    # 3. 应用饱和处理，保证值在有效范围内
                    temp_data = torch.clamp(temp_data, 0.0, 1.0)
                    
                    # 将处理后的数据复制回data_copy
                    data_copy[idx] = temp_data
        
        return data_copy
    
    def inject_trigger_with_inference(self, data, attack_flags=None, raw_data=None, top_model=None, bottom_models=None):
        """基于标签推断选择性地注入触发器
        
        Args:
            data: 输入数据
            attack_flags: 初始攻击标志，可以是Tensor或列表
            raw_data: 原始输入数据
            top_model: 顶部模型
            bottom_models: 底部模型列表
        
        Returns:
            包含触发器的数据
        """
        batch_size = data.size(0)
        
        # 确保attack_flags是Tensor类型
        if attack_flags is not None and not isinstance(attack_flags, torch.Tensor):
            attack_flags = torch.tensor(attack_flags, dtype=torch.bool, device=data.device)
        
        # 如果没有推断模块或未初始化，使用原始触发器注入方法
        if self.label_inference is None or not self.label_inference.initialized:
            return self.inject_trigger(data, attack_flags)
        
        # 使用标签推断确定哪些样本应该被攻击
        inferred_labels, confidences = self.label_inference.infer_labels(
            data.view(batch_size, -1), top_model, bottom_models, raw_data
        )
        
        if inferred_labels is None:
            # 推断失败，使用原始攻击标志
            return self.inject_trigger(data, attack_flags)
        
        # 简化标签推断逻辑，确保我们攻击足够多的样本
        # 创建基于推断的攻击标志 - 攻击所有非目标类样本
        inference_attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=data.device)
        
        for i in range(batch_size):
            # 简化条件: 如果标签不是目标类，我们就攻击
            # 不再使用复杂的条件和置信度判断
            if attack_flags is not None and attack_flags[i] and inferred_labels[i] != self.target_class:
                inference_attack_flags[i] = True
            # 备选条件: 如果预测的标签是我们的源类别(如果定义了)，我们会优先攻击
            elif attack_flags is not None and attack_flags[i] and hasattr(self.args, 'source_class') and inferred_labels[i] == self.args.source_class:
                inference_attack_flags[i] = True
        
        # 如果标签推断未能识别任何样本（很少见），退回到原始标志
        if not inference_attack_flags.any() and attack_flags is not None and attack_flags.any():
            print("警告：标签推断没有识别到任何可攻击样本，使用原始攻击标志")
            return self.inject_trigger(data, attack_flags)
        
        # 应用触发器 - 使用超强触发模式确保明显
        return self.inject_trigger(data, inference_attack_flags)

class CIFARBottomModel(nn.Module):
    """CIFAR底部模型，支持标准VILLAIN攻击"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(CIFARBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 使用ResNet风格的特征提取器，但用GroupNorm替代BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)  # 每组4个通道
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # 计算特征图大小
        feature_dim = 128 * (32 // 4) * (32 // 4)  # 经过2次stride=2的下采样
        
        # 分类器，使用GroupNorm替代BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GroupNorm(8, 512),  # 每组64个通道
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
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
        layers.append(ResBlock(in_channels, out_channels, stride, use_gn=True))
        # 其余块不需要下采样
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, use_gn=True))
        return nn.Sequential(*layers)
    
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
        
        # 特征提取 - 使用非inplace操作
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (8, 8))
        
        # 展平
        x_flat = x.view(x.size(0), -1)
        
        # 分类
        feat = self.classifier(x_flat)
        
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

class CIFARTopModel(nn.Module):
    """CIFAR top model，使用GroupNorm替代BatchNorm"""
    def __init__(self, input_dim=256, num_classes=10):
        super(CIFARTopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.gn1 = nn.GroupNorm(8, 512)  # 每组64个通道
        self.fc2 = nn.Linear(512, 256)
        self.gn2 = nn.GroupNorm(8, 256)  # 每组32个通道
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.gn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_epoch(modelC, bottom_models, optimizers, optimizerC, train_loader, epoch, args, defense_hooks):
    args_dict = vars(args)
    
    # 如果是 DPSGD 防御，使用 VFLSystem
    if (args_dict.get('defense_type', '').upper() == "DPSGD" and 
        hasattr(defense_hooks, 'instance') and defense_hooks.instance is not None):
        vfl_system = defense_hooks.instance
        vfl_system.train()
        
        # 获取 BatchMemoryManager
        if not hasattr(defense_hooks, 'batch_memory_manager'):
            raise ValueError("BatchMemoryManager not found in defense_hooks")
        batch_memory_manager = defense_hooks.batch_memory_manager
        
        # 获取优化器
        if not hasattr(defense_hooks, 'optimizer'):
            raise ValueError("Optimizer not found in defense_hooks")
        optimizer = defense_hooks.optimizer
    else:
        modelC.train()
        for model in bottom_models:
            model.train()
        batch_memory_manager = None
        optimizer = None
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        
        # 使用 BatchMemoryManager 处理批次
        if batch_memory_manager is not None:
            with batch_memory_manager:
                # 初始化梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = vfl_system(data)
                
                # 计算损失
                loss = F.cross_entropy(output, target)
                
                # 反向传播
                loss.backward()
        
                # 更新参数
                optimizer.step()
        else:
            # 初始化梯度
            for opt in optimizers:
                if opt is not None:
                    opt.zero_grad()
            if optimizerC is not None:
                optimizerC.zero_grad()
        
            # 前向传播
            bottom_outputs = []
            for model in bottom_models:
                output = model(data)
                bottom_outputs.append(output)
            feats = torch.cat(bottom_outputs, dim=1)
            # # ----【新增，最小侵入 DCT 过滤】----
            # if getattr(args, 'defense_type', 'NONE').upper() == 'MY':
            #     clean_feats, kept_idx, removed_idx, poison_mask = dct_trigger_filter(
            #         feats,
            #         tau=args.tau,
            #         k_min=args.k_min
            #     )
            #     feats = clean_feats
            #     target = target[kept_idx]  # 标签同步删除filter掉的
            
            output = modelC(feats)
            
            # 计算损失
            loss = F.cross_entropy(output, target)
            
            # 反向传播
            loss.backward()
            
            # 立即更新参数
            for opt in optimizers:
                if opt is not None:
                    opt.step()
            if optimizerC is not None:
                optimizerC.step()
        
        # 更新统计信息
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 打印进度
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None, defense_hooks=None):
    # 如果是 DPSGD 防御，使用 VFLSystem
    if (args.defense_type.upper() == "DPSGD" and 
        hasattr(defense_hooks, 'instance') and defense_hooks.instance is not None):
        vfl_system = defense_hooks.instance
        vfl_system.eval()
    else:
        modelC.eval()
    for i in range(len(bottom_models)):
        bottom_models[i].eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            # 如果是后门测试，注入触发器
            if is_backdoor:
                data, target, _attack_flags = prepare_backdoor_data(data, target, args)
            
            # 前向传播getattr(args, 'defense_type', 'NONE')
            if (args.defense_type.upper() == "DPSGD" and 
                hasattr(defense_hooks, 'instance') and defense_hooks.instance is not None):
                # 使用 VFLSystem 进行前向传播
                output = defense_hooks.instance(data)
            else:
                # 使用原有的分离模型进行前向传播
                bottom_outputs = []
                for i in range(len(bottom_models)):
                    output = bottom_models[i](data)
                    
                    # 应用防御
                    if hasattr(defense_hooks, 'bottom_defenses') and i < len(defense_hooks.bottom_defenses):
                        if hasattr(defense_hooks.bottom_defenses[i], 'forward') and defense_hooks.bottom_defenses[i].forward:
                            output = defense_hooks.bottom_defenses[i].forward(output)
                    
                    bottom_outputs.append(output)
                
                # 合并特征
                combined_output = torch.cat(bottom_outputs, dim=1)
                
                # 对聚合特征应用防御
                if hasattr(defense_hooks, 'forward') and defense_hooks.forward:
                    combined_output = defense_hooks.forward(combined_output)
                    
                # ----【新增，最小侵入 DCT 过滤】----
                if getattr(args, 'defense_type', 'NONE').upper() == 'MY':
                    clean_feats, kept_idx, removed_idx, poison_mask = dct_trigger_filter(
                        combined_output,
                        tau=args.tau,
                        k_min=args.k_min
                    )
                    combined_output = clean_feats
                    target = target[kept_idx]  # 标签同步删除filter掉的
                
                # 顶部模型
                output = modelC(combined_output)
            
            # 计算损失
            test_loss += F.cross_entropy(output, target).item()
            
            # 统计
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def load_dataset(dataset_name, data_dir, batch_size):
    """加载CIFAR数据集 - 支持自动下载和详细进度显示"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    print("\n2. 检查CIFAR数据集路径...")
    data_root = data_dir
    
    # 确保数据根目录存在
    os.makedirs(data_root, exist_ok=True)
    print(f"数据根目录: {data_root}")
    
    # 检查数据集文件是否已存在
    cifar10_exists = os.path.exists(os.path.join(data_root, 'cifar-10-batches-py'))
    cifar100_exists = os.path.exists(os.path.join(data_root, 'cifar-100-python'))
    
    print("\n3. 加载CIFAR数据集...")
    try:
        if dataset_name.upper() == 'CIFAR10':
            if cifar10_exists:
                print("找到已有的CIFAR-10数据集，直接加载...")
            else:
                print("CIFAR-10数据集不存在，开始自动下载...")
                print("数据集大小: ~170MB")
                print("下载可能需要几分钟，请耐心等待...")
            
            # 加载训练集（会自动下载）
            print("正在加载训练集...")
            train_dataset = datasets.CIFAR10(
                root=data_root, train=True, download=not cifar10_exists, transform=transform_train
            )
            
            # 加载测试集
            print("正在加载测试集...")
            test_dataset = datasets.CIFAR10(
                root=data_root, train=False, download=False, transform=transform_test
            )
            num_classes = 10
            
        elif dataset_name.upper() == 'CIFAR100':
            if cifar100_exists:
                print("找到已有的CIFAR-100数据集，直接加载...")
            else:
                print("CIFAR-100数据集不存在，开始自动下载...")
                print("数据集大小: ~170MB")
                print("下载可能需要几分钟，请耐心等待...")
            
            # 加载训练集（会自动下载）
            print("正在加载训练集...")
            train_dataset = datasets.CIFAR100(
                root=data_root, train=True, download=not cifar100_exists, transform=transform_train
            )
            
            # 加载测试集
            print("正在加载测试集...")
            test_dataset = datasets.CIFAR100(
                root=data_root, train=False, download=False, transform=transform_test
            )
            num_classes = 100
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}，支持的数据集: CIFAR10, CIFAR100")
            
        print(f"{dataset_name} 数据集加载成功!")
        
        # 验证数据集完整性
        print("验证数据集完整性...")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("数据集为空，可能下载不完整")
        
        # 检查类别数量
        if hasattr(train_dataset, 'classes'):
            actual_classes = len(train_dataset.classes)
            if actual_classes != num_classes:
                print(f"警告: 期望{num_classes}个类别，实际找到{actual_classes}个类别")
            else:
                print(f"类别验证通过: {num_classes}个类别")
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        
        # 详细的错误诊断
        if "Connection" in str(e) or "HTTP" in str(e) or "timeout" in str(e).lower():
            print("\n可能的解决方案:")
            print("1. 检查网络连接")
            print("2. 稍后重试")
            print("3. 手动下载数据集:")
            if dataset_name.upper() == 'CIFAR10':
                print("   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
            else:
                print("   - CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
            print(f"   - 解压到: {data_root}")
        elif "Permission" in str(e) or "access" in str(e).lower():
            print(f"\n权限错误: 无法写入 {data_root}")
            print("请检查目录权限或使用其他路径")
        elif "Space" in str(e) or "disk" in str(e).lower():
            print(f"\n磁盘空间不足: {data_root}")
            print("请清理磁盘空间或使用其他路径")
        else:
            print("\n未知错误，请检查数据集路径是否正确，以及是否有读写权限")
        
        sys.exit(1)
    
    print("\n4. 创建数据加载器...")
    
    # 优化数据加载参数 - 减少多进程开销
    num_workers = 2 if torch.cuda.is_available() else 0  # 从4降到2
    pin_memory = False  # 禁用pin_memory避免潜在问题
    
    print(f"数据加载配置:")
    print(f"- workers: {num_workers}")
    print(f"- pin_memory: {pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 避免最后一个batch大小不一致的问题
        persistent_workers=False  # 避免持久化worker
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
    print(f"类别数量: {num_classes}")
    
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
    
    return train_loader, test_loader, num_classes

# 添加load_data函数作为load_dataset的别名
load_data = load_dataset

def create_models(args):
    """创建模型"""
    output_dim = 64  # 每个底部模型的输出维度
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        if i == args.bkd_adversary:
            # 创建恶意模型
            model = CIFARBottomModel(
                input_dim=3,  # RGB图像
                output_dim=output_dim,
                is_adversary=True,
                args=args
            )
        else:
            # 创建正常模型
            model = CIFARBottomModel(
                input_dim=3,
                output_dim=output_dim
            )
        model = model.to(DEVICE)  # 确保模型在GPU上
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = CIFARTopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    ).to(DEVICE)  # 确保模型在GPU上
    
    # 创建并设置BadVFL触发器
    badvfl_trigger = BadVFLTrigger(args)
    bottom_models[args.bkd_adversary].set_badvfl_trigger(badvfl_trigger)
    
    return bottom_models, modelC

def prepare_backdoor_data(data, target, args):
    """准备后门数据，注入后门触发器 - 修复强制攻击2/3样本的问题"""
    batch_size = data.size(0)
    
    # 严格按照毒化预算设置攻击样本数量 - 移除强制2/3攻击的逻辑
    attack_portion = int(batch_size * args.poison_budget)
    attack_portion = min(attack_portion, batch_size)  # 确保不超过batch大小
    
    # 设置攻击标志
    dev = target.device
    attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=dev)
    
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
    
    # 调试信息 - 降低打印频率
    if random.random() < 0.01:  # 从5%降低到1%
        print(f"[BadVFL] 批次大小: {batch_size}, 毒化预算: {args.poison_budget:.2f}, 实际攻击样本: {attack_flags.sum().item()}")
    
    return bkd_data, bkd_target, attack_flags

def collect_inference_data(modelC, bottom_models, train_loader, args):
    """收集标签推断数据，使用BadVFL的标签推断方法 - 优化性能版本"""
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
    
    # 大幅减少收集批次数量，快速初始化
    max_batches = 50  # 从600大幅降到50
    
    print(f"将收集最多 {max_batches} 个批次的梯度数据用于标签推断 (快速模式)")
    
    # 检查是否是DPSGD防御
    defense_type = getattr(args, 'defense_type', 'NONE')
    is_dpsgd = defense_type.upper() == "DPSGD"
    
    if is_dpsgd:
        print("检测到DPSGD防御，跳过标签推断数据收集阶段")
        # 对于DPSGD，我们简单地初始化标签推断模块而不收集真实梯度
        # 创建一些假数据来初始化模块
        dummy_features = torch.randn(100, 3072).to(DEVICE)  # 3072 = 3*32*32 for CIFAR
        dummy_labels = torch.randint(0, args.num_classes, (100,)).to(DEVICE)
        
        # 更新标签推断历史
        samples_added = label_inference_module.update_with_batch(dummy_features, dummy_labels)
        
        # 尝试初始化
        if label_inference_module.embedding_swapping():
            if label_inference_module.candidate_selection():
                if adversary_model.badvfl_trigger:
                    adversary_model.badvfl_trigger.update_inference_stats()
                    print("DPSGD模式下标签推断初始化成功!")
                    return True
        
        print("DPSGD模式下标签推断初始化失败，但继续训练")
        return False
    
    # 非DPSGD情况，使用原来的收集方法
    # 创建优化器（仅用于梯度计算）
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
            samples_added = label_inference_module.update_with_batch(original_data, saved_grad)
            
            # 每10个批次尝试一次初始化，而不是每个批次都尝试
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
    
    # 最终尝试执行嵌入交换和候选选择
    print(f"最终尝试标签推断初始化，已收集 {len(label_inference_module.history_features)} 个样本")
    
    if label_inference_module.embedding_swapping():
        if label_inference_module.candidate_selection():
            # 更新触发器状态
            if adversary_model.badvfl_trigger:
                adversary_model.badvfl_trigger.update_inference_stats()
                return True
    
    return False

# 添加ResBlock的定义
class ResBlock(nn.Module):
    """ResNet基本块，使用GroupNorm替代BatchNorm"""
    def __init__(self, in_channels, out_channels, stride=1, use_gn=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
        
        # 如果需要下采样，或者输入输出通道数不一致，则使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 主路径
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        # 快捷路径
        shortcut = self.shortcut(x)
        
        # 合并主路径和快捷路径
        out = out + shortcut
        out = self.relu(out)
        
        return out

class DefenseWrapper(nn.Module):
    def __init__(self, bottom_models, top_model, defense_type="NONE", args=None, train_loader=None, optimizer=None):
        super(DefenseWrapper, self).__init__()
        self.bottom_models = bottom_models
        self.top_model = top_model
        self.defense_type = defense_type
        self.args = args
        self.train_loader = train_loader
        self.optimizer = optimizer  # 新增 optimizer 参数
        self._init_defense()
    
    def _init_defense(self):
        args_dict = vars(self.args).copy()
        if 'defense_type' in args_dict:
            del args_dict['defense_type']
        _, _, self.defense_hooks = build_defense(
            model=self.top_model,
            optimizer=self.optimizer,
                defense_type=self.defense_type,
            train_loader=self.train_loader,  # 传入 train_loader 参数
            **args_dict  # 将 self.args 转换为字典，并移除 defense_type 参数
        )

        # 如果是 DPSGD 防御，确保 optimizer 已正确设置
        if self.defense_type.upper() == "DPSGD":
            if self.defense_hooks.instance:
                self.optimizer = self.defense_hooks.instance.optimizer

def save_checkpoint(modelC, bottom_models, epoch, clean_acc, asr=None, inference_acc=None):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 使用更结构化的方式命名
    defense_str = f"defense_{args.defense_type}" if args.defense_type.upper() != 'NONE' else "no_defense"
    attack_str = "badvfl"
    
    model_name = f"{args.dataset}_{attack_str}_{defense_str}_party_{args.party_num}_bkd_adv_{args.bkd_adversary}"
    
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
        'args': args
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存模型到 {model_save_path}')

class VFLSystem(nn.Module):
    def __init__(self, bottom_models, top_model):
        super(VFLSystem, self).__init__()
        self.bottom_models = nn.ModuleList(bottom_models)
        self.top_model = top_model
    
    def forward(self, x):
        # 前向传播，分别处理每个bottom model
        bottom_outputs = []
        for model in self.bottom_models:
            output = model(x)
            bottom_outputs.append(output)
        
        # 合并特征
        feats = torch.cat(bottom_outputs, dim=1)
        
        # 顶层模型前向传播
        output = self.top_model(feats)
        return output

def main():
    """主训练循环 - 完整的BadVFL攻击实现"""
    # 使用全局的 args 变量（已在文件开头解析）
    set_random_seed(args.seed)
    
    # 参数验证和优化
    defense_type = getattr(args, 'defense_type', 'NONE')
    print(f"Defense type: {defense_type}")
    
    if defense_type.upper() != 'NONE':
        # 确保防御参数合理
        if defense_type.upper() == 'DPSGD':
            args.dpsgd_noise_multiplier = max(0.5, min(1.0, args.dpsgd_noise_multiplier))
            args.dpsgd_max_grad_norm = max(0.8, min(1.2, args.dpsgd_max_grad_norm))
            args.dpsgd_epsilon = max(3.0, min(10.0, args.dpsgd_epsilon))
        elif defense_type.upper() == 'MP':
            args.mp_pruning_amount = max(0.2, min(0.5, args.mp_pruning_amount))
        elif defense_type.upper() == 'ANP':
            args.anp_sigma = max(0.1, min(0.4, args.anp_sigma))
        elif defense_type.upper() == 'BDT':
            args.bdt_prune_ratio = max(0.2, min(0.5, args.bdt_prune_ratio))
        elif defense_type.upper() == 'VFLIP':
            args.vflip_threshold = max(1.2, min(3.0, args.vflip_threshold))
        elif defense_type.upper() == 'ISO':
            args.iso_lr = max(0.0001, min(0.01, args.iso_lr))
        
        # 调整攻击参数以适应防御
        args.backdoor_weight = max(0.5, min(1.0, args.backdoor_weight))
        args.poison_budget = max(0.05, min(0.15, args.poison_budget))
        args.trigger_intensity = max(0.6, min(0.8, args.trigger_intensity))
    
    # 加载数据集
    train_loader, test_loader, num_classes = load_data(args.dataset, args.data_dir, args.batch_size)
    args.num_classes = num_classes
    
    # 创建模型
    bottom_models, modelC = create_models(args)
    
    # 初始化优化器
    optimizers = []
    for model in bottom_models:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizers.append(optimizer)
    optimizerC = optim.SGD(modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 初始化防御
    if defense_type.upper() == "DPSGD":
        print("Initializing DPSGD defense...")
        
        # 创建 VFLSystem
        vfl_system = VFLSystem(bottom_models, modelC)
            
        # 创建优化器
        optimizer = optim.SGD(vfl_system.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        # 使用 PrivacyEngine 包装模型和优化器
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
            
        vfl_system, optimizer, train_loader = privacy_engine.make_private(
            module=vfl_system,
            optimizer=optimizer,
            data_loader=train_loader,
                noise_multiplier=args.dpsgd_noise_multiplier,
                max_grad_norm=args.dpsgd_max_grad_norm,
            )
            
        # 创建 BatchMemoryManager
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        
        # 设置防御钩子
        from types import SimpleNamespace
        defense_hooks = SimpleNamespace()
        defense_hooks.instance = vfl_system
        defense_hooks.optimizer = optimizer
        defense_hooks.privacy_engine = privacy_engine
        
        # 创建 BatchMemoryManager 作为上下文管理器使用
        defense_hooks.batch_memory_manager = BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.batch_size,
            optimizer=optimizer
        )
        
        print("DPSGD defense initialized with PrivacyEngine and BatchMemoryManager")
    
    else:
        # 非 DPSGD 防御的处理
        defense_wrapper = DefenseWrapper(bottom_models, modelC, defense_type, args, train_loader, optimizerC)
        defense_hooks = defense_wrapper.defense_hooks

    # 打印训练配置
    print(f"\n{'='*50}")
    print(f"BadVFL攻击训练配置")
    print(f"{'='*50}")
    print(f"数据集: {args.dataset}")
    print(f"参与方数量: {args.party_num}")
    print(f"恶意方ID: {args.bkd_adversary}")
    print(f"目标类别: {args.target_class}")
    print(f"后门注入开始轮次: {args.Ebkd}")
    print(f"毒化预算: {args.poison_budget}")
    print(f"触发器强度: {args.trigger_intensity}")
    print(f"后门权重: {args.backdoor_weight}")
    print(f"防御类型: {defense_type}")
    print(f"{'='*50}")
    
    # 获取恶意模型用于标签推断
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference
    
    # 阶段1: 标签推断数据收集阶段（在Ebkd之前）
    print(f"\n{'='*20} 标签推断数据收集阶段 (1-{args.Ebkd-1}轮) {'='*20}")
    
    # 预训练阶段，收集标签推断数据
    if args.Ebkd > 1:
        # 进行预训练，收集梯度用于标签推断
        inference_initialized = False
        
        # 对于DPSGD防御，我们在预训练阶段不使用DPSGD，只在攻击阶段使用
        temp_defense_type = defense_type
        if defense_type.upper() == "DPSGD":
            print("预训练阶段暂时禁用DPSGD防御，使用正常训练模式收集标签推断数据")
            # 创建临时的无防御hooks
            temp_defense_hooks = SimpleNamespace()
            temp_defense_hooks.forward = None
            temp_defense_hooks.instance = None  # 确保没有VFLSystem实例
            
            # 对于DPSGD，我们需要使用原始的模型和优化器进行预训练
            # 重新创建未被Opacus包装的模型
            temp_bottom_models, temp_modelC = create_models(args)
            
            # 重新创建优化器
            temp_optimizers = []
            for model in temp_bottom_models:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                temp_optimizers.append(optimizer)
            temp_optimizerC = optim.SGD(temp_modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            
        else:
            temp_defense_hooks = defense_hooks
            temp_bottom_models = bottom_models
            temp_modelC = modelC
            temp_optimizers = optimizers
            temp_optimizerC = optimizerC
        
        for epoch in range(min(args.Ebkd, 5)):  # 只在前几个epoch收集数据
            print(f"\n--- 预训练Epoch {epoch} (收集标签推断数据) ---")
            
            # 正常训练
            train_loss, train_acc = train_epoch(
                temp_modelC, temp_bottom_models, temp_optimizers, temp_optimizerC,
                train_loader, epoch, args, temp_defense_hooks
            )
            
            # 测试
            test_loss, test_acc = test(
                temp_modelC, temp_bottom_models, test_loader,
                is_backdoor=False, epoch=epoch, args=args, defense_hooks=temp_defense_hooks
            )
            
            print(f"Epoch {epoch} - 预训练:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # 收集标签推断数据
            if epoch >= 2 and not inference_initialized:  # 从第3个epoch开始收集
                print(f"开始收集标签推断数据...")
                inference_initialized = collect_inference_data(temp_modelC, temp_bottom_models, train_loader, args)
                if inference_initialized:
                    print(f"标签推断初始化成功!")
                    break
        
        # 如果使用DPSGD防御，现在将预训练的权重复制到DPSGD模型
        if defense_type.upper() == "DPSGD":
            print("将预训练权重复制到DPSGD模型...")
            
            # 复制权重到原始模型
            for i, (temp_model, orig_model) in enumerate(zip(temp_bottom_models, bottom_models)):
                orig_model.load_state_dict(temp_model.state_dict())
            modelC.load_state_dict(temp_modelC.state_dict())
            
            print("重新初始化DPSGD防御系统用于攻击阶段...")
            
            # 重新创建 VFLSystem
            vfl_system = VFLSystem(bottom_models, modelC)
            
            # 重新创建优化器
            optimizer = optim.SGD(vfl_system.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            
            # 重新使用 PrivacyEngine 包装模型和优化器
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            
            vfl_system, optimizer, train_loader = privacy_engine.make_private(
                module=vfl_system,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.dpsgd_noise_multiplier,
                max_grad_norm=args.dpsgd_max_grad_norm,
            )
            
            # 重新创建 BatchMemoryManager
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            
            # 更新防御钩子
            defense_hooks.instance = vfl_system
            defense_hooks.optimizer = optimizer
            defense_hooks.privacy_engine = privacy_engine
            
            # 重新创建 BatchMemoryManager
            defense_hooks.batch_memory_manager = BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=args.batch_size,
                optimizer=optimizer
            )
            
            print("DPSGD防御系统重新初始化完成")
            
            # 将标签推断模块从临时模型转移到新模型
            if hasattr(temp_bottom_models[args.bkd_adversary], 'label_inference'):
                bottom_models[args.bkd_adversary].label_inference = temp_bottom_models[args.bkd_adversary].label_inference
                if hasattr(temp_bottom_models[args.bkd_adversary], 'badvfl_trigger'):
                    bottom_models[args.bkd_adversary].badvfl_trigger = temp_bottom_models[args.bkd_adversary].badvfl_trigger
                    bottom_models[args.bkd_adversary].badvfl_trigger.set_label_inference(bottom_models[args.bkd_adversary].label_inference)
    
    else:
        # 如果Ebkd为1或0，直接尝试初始化标签推断
        inference_initialized = collect_inference_data(modelC, bottom_models, train_loader, args)
    
    # 阶段2: BadVFL攻击训练阶段（从Ebkd开始）
    print(f"\n{'='*20} BadVFL攻击阶段 (从第{args.Ebkd}轮开始) {'='*20}")
    
    best_clean_acc = 0
    best_asr = 0
    inference_acc_history = []
    
    # 主训练循环
    for epoch in range(args.epochs):
        # 判断是否在攻击阶段
        apply_backdoor = epoch >= args.Ebkd
        
        if apply_backdoor:
            print(f"\n--- BadVFL攻击 Epoch {epoch} ---")
        else:
            print(f"\n--- 正常训练 Epoch {epoch} ---")
        
        # 训练
        if apply_backdoor:
            # 攻击训练
            train_loss, train_acc = train_epoch_with_attack(
                modelC, bottom_models, optimizers, optimizerC,
                train_loader, epoch, args, defense_hooks
            )
        else:
            # 正常训练
            train_loss, train_acc = train_epoch(
                modelC, bottom_models, optimizers, optimizerC,
                train_loader, epoch, args, defense_hooks
            )
        
        # 测试干净样本
        test_loss, test_acc = test(
            modelC, bottom_models, test_loader,
            is_backdoor=False, epoch=epoch, args=args, defense_hooks=defense_hooks
            )
        
        # 后门测试（只在攻击阶段）
        asr = 0
        if apply_backdoor:
            try:
                # 创建后门测试集
                backdoor_test_loader = create_backdoor_test_loader(test_loader, args)
                
                # 测试后门攻击成功率
                backdoor_loss, asr = test(
                    modelC, bottom_models, backdoor_test_loader,
                    is_backdoor=True, epoch=epoch, args=args, defense_hooks=defense_hooks
            )
        
                print(f"  后门测试 - Loss: {backdoor_loss:.4f}, ASR: {asr:.2f}%")
            except Exception as e:
                print(f"  后门测试失败: {e}")
                asr = 0
        
        # 标签推断准确率测试
        inference_acc = 0
        if label_inference_module and label_inference_module.initialized:
            try:
                inference_acc = test_label_inference(
                    modelC, bottom_models, test_loader, args
                )
                inference_acc_history.append(inference_acc)
                print(f"  标签推断准确率: {inference_acc:.2f}%")
            except Exception as e:
                print(f"  标签推断测试失败: {e}")
        
        # 打印结果总结
        print(f"\nEpoch {epoch} Summary:")
        print(f"  - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if apply_backdoor:
            print(f"  - Attack Success Rate (ASR): {asr:.2f}%")
        if inference_acc > 0:
            print(f"  - Label Inference Acc: {inference_acc:.2f}%")
        
        # 更新最佳结果
        if test_acc > best_clean_acc:
            # save ckpt by best test acc
            best_clean_acc = test_acc
            save_checkpoint(
                modelC, bottom_models, epoch, test_acc, asr, inference_acc
            )
        if asr > best_asr:
            best_asr = asr
        
        # # 保存检查点
        # if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        #     save_checkpoint(
        #         modelC, bottom_models, epoch, test_acc, asr, inference_acc
        #     )
    
    # 最终结果报告
    print(f"\n{'='*50}")
    print(f"BadVFL攻击训练完成!")
    print(f"{'='*50}")
    print(f"最佳干净准确率: {best_clean_acc:.2f}%")
    print(f"最佳攻击成功率: {best_asr:.2f}%")
    if inference_acc_history:
        avg_inference_acc = sum(inference_acc_history) / len(inference_acc_history)
        print(f"平均标签推断准确率: {avg_inference_acc:.2f}%")
    print(f"防御类型: {defense_type}")
    print(f"{'='*50}")

def train_epoch_with_attack(modelC, bottom_models, optimizers, optimizerC, train_loader, epoch, args, defense_hooks):
    """带有BadVFL攻击的训练epoch"""
    args_dict = vars(args)
    
    # 如果是 DPSGD 防御，使用 VFLSystem
    if (args_dict.get('defense_type', '').upper() == "DPSGD" and 
        hasattr(defense_hooks, 'instance') and defense_hooks.instance is not None):
        vfl_system = defense_hooks.instance
        vfl_system.train()
        batch_memory_manager = defense_hooks.batch_memory_manager
        optimizer = defense_hooks.optimizer
    else:
        modelC.train()
        for model in bottom_models:
            model.train()
        batch_memory_manager = None
        optimizer = None
    
    total_loss = 0
    total_backdoor_loss = 0
    correct = 0
    total = 0
    attack_count = 0
    
    # 获取恶意模型和触发器
    adversary_model = bottom_models[args.bkd_adversary]
    badvfl_trigger = adversary_model.badvfl_trigger
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        batch_size = data.size(0)
        
        # 准备后门数据
        backdoor_data, backdoor_target, attack_flags = prepare_backdoor_data(data, target, args)
        attack_count += attack_flags.sum().item()
        
        # 注入BadVFL触发器
        if badvfl_trigger and attack_flags.any():
            # 使用带推断的触发器注入
            backdoor_data = badvfl_trigger.inject_trigger_with_inference(
                backdoor_data, attack_flags, data, modelC, bottom_models
            )
        
        # 使用 BatchMemoryManager 处理批次
        if batch_memory_manager is not None:
            with batch_memory_manager:
                # 初始化梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = vfl_system(backdoor_data)
                
                # 计算损失
                clean_loss = F.cross_entropy(output, target)
                backdoor_loss = F.cross_entropy(output, backdoor_target)
                
                # 组合损失
                if attack_flags.any():
                    # 计算攻击权重
                    attack_weight = args.backdoor_weight
                    if epoch < args.Ebkd + 5:
                        # 逐渐增加攻击权重
                        attack_weight *= (epoch - args.Ebkd + 1) / 5.0
                    
                    # 混合损失
                    loss = clean_loss + attack_weight * backdoor_loss
                else:
                    loss = clean_loss
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                optimizer.step()
        else:
            # 初始化梯度
            for opt in optimizers:
                if opt is not None:
                    opt.zero_grad()
            if optimizerC is not None:
                optimizerC.zero_grad()
            
            # 前向传播
            bottom_outputs = []
            for model in bottom_models:
                output = model(backdoor_data, attack_flags if model.is_adversary else None)
                bottom_outputs.append(output)
            feats = torch.cat(bottom_outputs, dim=1)
            
            # ----【新增，最小侵入 DCT 过滤】----
            if getattr(args, 'defense_type', 'NONE').upper() == 'MY':
                clean_feats, kept_idx, removed_idx, poison_mask = dct_trigger_filter(
                        feats,
                        tau=args.tau,
                        k_min=args.k_min
                    )
                feats = clean_feats
                target = target[kept_idx]  # 标签同步删除filter掉的
                backdoor_target = backdoor_target[kept_idx]  # 标签同步删除filter掉的
            
            output = modelC(feats)
            
            # 计算损失
            clean_loss = F.cross_entropy(output, target)
            backdoor_loss = F.cross_entropy(output, backdoor_target)
            
            # 组合损失
            if attack_flags.any():
                # 计算攻击权重
                attack_weight = args.backdoor_weight
                if epoch < args.Ebkd + 5:
                    # 逐渐增加攻击权重
                    attack_weight *= (epoch - args.Ebkd + 1) / 5.0
                
                # 混合损失
                loss = clean_loss + attack_weight * backdoor_loss
            else:
                loss = clean_loss
            
            # 反向传播
            loss.backward()
            
            # 立即更新参数
            for opt in optimizers:
                if opt is not None:
                    opt.step()
            if optimizerC is not None:
                optimizerC.step()
        
        # 更新统计信息
        total_loss += clean_loss.item()
        total_backdoor_loss += backdoor_loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(backdoor_target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 打印进度
        if batch_idx % args.log_interval == 0:
            print(f'BadVFL Attack Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Clean Loss: {clean_loss.item():.6f}\t'
                  f'Backdoor Loss: {backdoor_loss.item():.6f}\t'
                  f'Attack Samples: {attack_flags.sum().item()}/{batch_size}')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    avg_backdoor_loss = total_backdoor_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'BadVFL Attack Epoch: {epoch}\t'
          f'Avg Clean Loss: {avg_loss:.6f}\t'
          f'Avg Backdoor Loss: {avg_backdoor_loss:.6f}\t'
          f'Accuracy: {accuracy:.2f}%\t'
          f'Total Attack Samples: {attack_count}')
    
    return avg_loss, accuracy

def create_backdoor_test_loader(test_loader, args):
    """创建后门测试数据加载器"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # 收集测试数据
    all_data = []
    all_targets = []
    
    for data, target in test_loader:
        all_data.append(data)
        all_targets.append(target)
    
    # 合并所有数据
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 只选择非目标类的样本用于后门测试
    non_target_mask = all_targets != args.target_class
    test_data = all_data[non_target_mask]
    test_targets = all_targets[non_target_mask]
    
    # 限制测试样本数量
    max_test_samples = min(1000, len(test_data))
    test_data = test_data[:max_test_samples]
    test_targets = test_targets[:max_test_samples]
    
    # 准备后门数据
    backdoor_data, backdoor_targets, attack_flags = prepare_backdoor_data(
        test_data, test_targets, args
    )
    
    # 只保留被攻击的样本
    attacked_data = backdoor_data[attack_flags]
    attacked_targets = backdoor_targets[attack_flags]
    
    # 创建数据集和数据加载器
    backdoor_dataset = TensorDataset(attacked_data, attacked_targets)
    backdoor_test_loader = DataLoader(
        backdoor_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    return backdoor_test_loader

def test_label_inference(modelC, bottom_models, test_loader, args):
    """测试标签推断准确率"""
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference
    
    if not label_inference_module or not label_inference_module.initialized:
        return 0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(args.device)
            target = target.to(args.device)
            
            # 使用标签推断模块预测标签
            inferred_labels, confidence = label_inference_module.infer_labels(
                data.view(data.size(0), -1), modelC, bottom_models, data
            )
            
            if inferred_labels is not None:
                inferred_labels = torch.tensor(inferred_labels, device=args.device)
                correct += (inferred_labels == target).sum().item()
                total += target.size(0)
            
            # 限制测试样本数量以节省时间
            if total >= 1000:
                break
    
    return 100. * correct / total if total > 0 else 0

if __name__ == '__main__':
    main()