#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: train_cinic10_badvfl_with_inference.py
# 描述: 针对CINIC10数据集的BadVFL攻击训练 (带标签推断)
# -----------------------------------------------------
# 此脚本基于 train_cifar_badvfl_with_inference.py 修改而来
# 主要改动:
# - 数据集加载和预处理: 使用CINIC10数据集
# - 模型架构: 可能需要微调以适应CINIC10的特性
# - 超参数: 针对CINIC10进行调整，例如学习率、后门权重等
# -----------------------------------------------------

import argparse
import os
import sys

# ===================================================================
# 1. 解析参数并设置GPU环境 (必须在导入torch之前完成)
# ===================================================================
parser = argparse.ArgumentParser(description='针对CINIC-10数据集的BadVFL攻击训练 (带标签推断)')
# 参数设置
parser.add_argument('--dataset', type=str, default='CINIC10', help='数据集名称')
parser.add_argument('--batch-size', type=int, default=128, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=float, default=2, help='BadVFL触发器大小 (用于兼容性)')
parser.add_argument('--pattern-size', type=float, default=2, help='BadVFL触发器大小 (与trigger-size相同)')
parser.add_argument('--trigger-intensity', type=float, default=1.0, help='BadVFL触发器强度')  # 降低默认触发器强度
parser.add_argument('--position', type=str, default='dr', help='触发器位置 (dr=右下, ul=左上, mid=中间, ml=中左)')
parser.add_argument('--gpu', type=int, default=0, help='要使用的物理GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=15, help='早停轮数')
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
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_badvfl_cinic10', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量 (10)')
parser.add_argument('--data-dir', type=str, default='/home/steve/VFLIP-esorics24-master/data/CINIC10/', help='数据集目录')
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
                    help='Defense type (NONE, DPSGD, MP, ANP, BDT, VFLIP, ISO)')
# DPSGD args
parser.add_argument('--dpsgd-noise-multiplier', type=float, default=1.0, help='Noise multiplier for DPSGD')
parser.add_argument('--dpsgd-max-grad-norm', type=float, default=1.0, help='Max grad norm for DPSGD')
parser.add_argument('--dpsgd-epsilon', type=float, default=10.0, help='Privacy budget epsilon for DPSGD')
# MP args
parser.add_argument('--mp-pruning-amount', type=float, default=0.2, help='Pruning amount for MP')
# ANP args
parser.add_argument('--anp-sigma', type=float, default=0.1, help='Sigma for Gaussian noise in ANP')
# BDT args
parser.add_argument('--bdt-prune-ratio', type=float, default=0.2, help='Prune ratio for BDT')
# VFLIP args
parser.add_argument('--vflip-threshold', type=float, default=3.0, help='Anomaly threshold for VFLIP')
parser.add_argument('--vflip-train-epochs', type=int, default=5, help='Number of epochs to pre-train VFLIP MAE')
# ISO args
parser.add_argument('--iso-lr', type=float, default=1e-3, help='Learning rate for ISO layer')

# 二元分类器参数
parser.add_argument('--binary-classifier', type=str, default='randomforest', choices=['randomforest', 'logistic'], 
                    help='二元分类器类型 (randomforest 或 logistic)')

# 早停参数
parser.add_argument('--early-stopping', action='store_true',
                    help='启用早停 (default: False)')
parser.add_argument('--monitor', type=str, default='test_acc', choices=['test_acc', 'inference_acc'],
                    help='监控指标，用于早停判断 (default: test_acc)')

# 源类别参数 
parser.add_argument('--source-class', type=int, default=6, help='攻击源类别')

# 解析参数
ARGS = parser.parse_args()

# 使用os.environ['CUDA_VISIBLE_DEVICES']来严格限制可见的GPU
# 这是最稳健的方法，可以防止PyTorch意外使用其他GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(ARGS.gpu)
print(f"关键设置: 通过 os.environ['CUDA_VISIBLE_DEVICES'] 将程序限制在物理 GPU {ARGS.gpu} 上")
print("现在，PyTorch只会看到一个GPU，并将其标识为 'cuda:0'")


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import time
import random
from tqdm import tqdm
from PIL import Image
import tarfile
import requests
from types import SimpleNamespace

from defense_all import build_defense
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
import math
import shutil
import urllib.request

# 全局变量
DEVICE = None
# ARGS 已在上面定义

# 设置全局变量和配置
# 因为设置了CUDA_VISIBLE_DEVICES，这里可以直接用'cuda'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 添加GPU内存监控函数
def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        return "GPU不可用"
    
    try:
        # 获取当前设备内存使用情况
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return f"GPU内存: 已分配={allocated:.1f}MB, 已保留={reserved:.1f}MB, 峰值={max_allocated:.1f}MB"
    except:
        return "无法获取GPU内存信息"

# 添加随机种子设置函数
def setup_seed(seed):
    """设置所有随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置cudnn为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

# 在程序开始处强制检查CUDA可用性
def check_cuda():
    """强制检查CUDA可用性并打印信息"""
    if torch.cuda.is_available():
        # 因为设置了CUDA_VISIBLE_DEVICES，所以只有一个设备可见
        print("\n==== CUDA可用 ====")
        print(f"设备数量: {torch.cuda.device_count()} (被CUDA_VISIBLE_DEVICES限制)")
        print(f"可见设备 'cuda:0': {torch.cuda.get_device_name(0)} (物理ID: {ARGS.gpu})")
        print(f"当前PyTorch设备: cuda:{torch.cuda.current_device()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print("=================\n")
        return True
    else:
        print("\n警告: CUDA不可用，将使用CPU\n")
        return False

# 扩展命令行参数 - 已移动到文件顶部

# 设置全局变量 - 已移动到文件顶部

# 参数后处理 - 确保pattern_size和trigger_size保持一致
if ARGS.trigger_size != ARGS.pattern_size:
    print(f"注意: 检测到trigger_size ({ARGS.trigger_size}) 和 pattern_size ({ARGS.pattern_size}) 不一致")
    print(f"为保持一致性，将使用trigger_size值 {ARGS.trigger_size}")
    ARGS.pattern_size = ARGS.trigger_size

# 强制检查CUDA可用性并设置设备 - 已通过 os.environ 完成，无需额外操作
if torch.cuda.is_available():
    print(f"PyTorch将自动使用可见的GPU: {torch.cuda.get_device_name(0)}")
else:
    print("警告: CUDA不可用，将使用CPU进行训练（性能会很慢）")


# 确保模型创建时明确指定设备
def create_models():
    """创建模型 - 重写版本，确保模型在正确的GPU上创建"""
    global DEVICE
    output_dim = 64  # 每个底部模型的输出维度
    
    # 确认当前设备 (现在总是 cuda:0)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()  # 总是返回 0
        print(f"当前CUDA设备: {current_device} - {torch.cuda.get_device_name(current_device)}")
    
    # 创建底部模型
    bottom_models = []
    for i in range(ARGS.party_num):
        if i == ARGS.bkd_adversary:
            # 创建恶意模型
            model = CINIC10BottomModel(
                input_dim=3,  # RGB图像
                output_dim=output_dim,
                is_adversary=True,
                args=ARGS
            )
        else:
            # 创建正常模型
            model = CINIC10BottomModel(
                input_dim=3,
                output_dim=output_dim
            )
        
        # 立即将模型移动到GPU
        if torch.cuda.is_available():
            model = model.to(DEVICE)
            print(f"模型 {i} 已移动到设备: {next(model.parameters()).device}")
        
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = CINIC10TopModel(
        input_dim=output_dim * ARGS.party_num,
        num_classes=ARGS.num_classes
    )
    
    if torch.cuda.is_available():
        modelC = modelC.to(DEVICE)
        print(f"顶部模型已移动到设备: {next(modelC.parameters()).device}")
    
    # 创建并设置BadVFL触发器
    badvfl_trigger = BadVFLTrigger(ARGS)
    if torch.cuda.is_available():
        # 确保触发器的pattern_mask也在正确的设备上
        if hasattr(badvfl_trigger, 'pattern_mask'):
            badvfl_trigger.pattern_mask = badvfl_trigger.pattern_mask.to(DEVICE)
            print(f"触发器模式已移动到: {badvfl_trigger.pattern_mask.device}")
            
    bottom_models[ARGS.bkd_adversary].set_badvfl_trigger(badvfl_trigger)
    
    # 验证所有模型都在正确的设备上
    if torch.cuda.is_available():
        print("\n验证模型设备位置:")
        for i, model in enumerate(bottom_models):
            try:
                device_str = next(model.parameters()).device
                print(f"模型 {i} 在设备: {device_str}")
                if str(device_str) != str(DEVICE):
                    print(f"警告: 模型 {i} 不在正确的设备上! 尝试重新移动...")
                    bottom_models[i] = bottom_models[i].to(DEVICE)
            except Exception as e:
                print(f"检查模型 {i} 失败: {str(e)}")
        
        try:
            device_str = next(modelC.parameters()).device
            print(f"顶部模型在设备: {device_str}")
            if str(device_str) != str(DEVICE):
                print(f"警告: 顶部模型不在正确的设备上! 尝试重新移动...")
                modelC = modelC.to(DEVICE)
        except Exception as e:
            print(f"检查顶部模型失败: {str(e)}")
    
    # 确保垃圾回收并释放CUDA缓存
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return bottom_models, modelC

# 标签推断器实现
class BadVFLLabelInference:
    """BadVFL攻击中的标签推断模块，直接使用模型输出进行推断"""
    def __init__(self, feature_dim, num_classes, args):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.args = args
        self.confidence_threshold = args.confidence_threshold
        
        # 存储历史推断结果和置信度 - 使用张量而非列表以提高效率
        self.history_features_cpu = []  # 仍保留CPU版本用于scikit-learn兼容
        self.history_predictions_cpu = []
        self.history_confidence_cpu = []
        
        # GPU版缓存 - 使用张量而非列表
        self.history_features_gpu = None
        self.history_predictions_gpu = None
        self.history_confidence_gpu = None
        
        # 辅助数据集用于训练推断模型
        self.auxiliary_dataset = None
        self.auxiliary_loader = None
        
        # 推断分类器 - 添加GPU版本的分类器
        self.inference_classifier = None  # 传统CPU分类器 (scikit-learn)
        self.inference_classifier_gpu = None  # GPU分类器 (PyTorch)
        self.initialized = False
        
        # 特征维度 - BadVFL使用完整图像作为特征
        self.expected_features = 3072  # 3×32×32 for CINIC-10 (与CIFAR相同)
        
        # 设置所需的最小样本数 - 降低所需的最小样本数以加快初始化
        self.min_samples = max(30, 4 * num_classes)  # 降低到只需要30个样本即可初始化
        
        # 创建GPU分类器 - PyTorch神经网络实现
        self._create_gpu_classifier()
        
        print(f"BadVFL标签推断模块创建: 特征维度={feature_dim}, 类别数={num_classes}")
        print(f"标签推断所需最小样本数: {self.min_samples}")
        print(f"已创建GPU版分类器，将使用设备: {DEVICE}")
    
    def _create_gpu_classifier(self):
        """创建GPU版分类器 - 使用PyTorch神经网络"""
        # 使用GPU友好的架构 - 增加中间层width并使用LeakyReLU激活函数提高非线性表达能力
        # 使用BatchNorm加速训练，使用Dropout防止过拟合
        self.inference_classifier_gpu = nn.Sequential(
            nn.Linear(self.expected_features, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 1)  # 移除sigmoid层，使用BCEWithLogitsLoss
        ).to(DEVICE)
        
        # 使用Kaiming初始化以加快收敛并提高GPU利用率
        for m in self.inference_classifier_gpu.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def update_with_batch(self, features, predictions, confidence=None):
        """更新特征和预测历史记录"""
        # 保存原始数据类型
        is_tensor = isinstance(features, torch.Tensor)
        
        # 为CPU版处理准备数据 - 仍保留用于兼容
        features_cpu = features.detach().cpu().numpy() if is_tensor else features
        predictions_cpu = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
            
        # 确保预测是离散标签，而不是连续值
        if len(predictions_cpu.shape) > 1:
            # 如果是多维数组，转换为类别索引
            if predictions_cpu.shape[1] > 1:
                predictions_cpu = np.argmax(predictions_cpu, axis=1)
            else:
                predictions_cpu = predictions_cpu.flatten()
        
        # 确保predictions_cpu是整数类型的标签
        if predictions_cpu.dtype != np.int64 and predictions_cpu.dtype != np.int32:
            predictions_cpu = predictions_cpu.astype(np.int32)
        
        if confidence is None:
            # 默认使用均匀置信度
            confidence_cpu = np.ones(len(features_cpu)) / self.num_classes
        elif isinstance(confidence, torch.Tensor):
            confidence_cpu = confidence.detach().cpu().numpy()
        else:
            confidence_cpu = confidence
        
        # 储存新的数据 - CPU版本
        self.history_features_cpu.extend(features_cpu)
        self.history_predictions_cpu.extend(predictions_cpu)
        self.history_confidence_cpu.extend(confidence_cpu)
        
        # 只保留最近的记录 - CPU版本
        max_history = self.args.history_size
        if len(self.history_features_cpu) > max_history:
            self.history_features_cpu = self.history_features_cpu[-max_history:]
            self.history_predictions_cpu = self.history_predictions_cpu[-max_history:]
            self.history_confidence_cpu = self.history_confidence_cpu[-max_history:]
        
        # 为GPU版处理准备数据 - 直接存储张量
        # 确保数据是张量并在GPU上
        if not is_tensor:
            features = torch.FloatTensor(features_cpu)
        
        # 确保张量在正确的设备上
        if features.device != DEVICE:
            features = features.to(DEVICE)
        
        # 创建或更新GPU历史记录
        if self.history_features_gpu is None:
            self.history_features_gpu = features
            # 创建预测和置信度张量
            self.history_predictions_gpu = torch.tensor(predictions_cpu, dtype=torch.float32, device=DEVICE)
            self.history_confidence_gpu = torch.tensor(confidence_cpu, dtype=torch.float32, device=DEVICE)
        else:
            # 添加新数据到GPU历史记录
            self.history_features_gpu = torch.cat([self.history_features_gpu, features], dim=0)
            
            # 添加预测和置信度
            new_predictions = torch.tensor(predictions_cpu, dtype=torch.float32, device=DEVICE)
            new_confidence = torch.tensor(confidence_cpu, dtype=torch.float32, device=DEVICE)
            
            self.history_predictions_gpu = torch.cat([self.history_predictions_gpu, new_predictions], dim=0)
            self.history_confidence_gpu = torch.cat([self.history_confidence_gpu, new_confidence], dim=0)
            
            # 控制GPU历史记录大小
            if len(self.history_features_gpu) > max_history:
                self.history_features_gpu = self.history_features_gpu[-max_history:]
                self.history_predictions_gpu = self.history_predictions_gpu[-max_history:]
                self.history_confidence_gpu = self.history_confidence_gpu[-max_history:]
        
        return len(features)
    
    def initialize_classifier(self):
        """初始化标签推断分类器，只使用GPU版本以提高效率"""
        # 首先检查样本数量是否足够
        if len(self.history_features_cpu) < self.min_samples:
            print(f"样本不足，无法初始化分类器: {len(self.history_features_cpu)}/{self.min_samples}")
            return False
        
        try:
            # 初始化PyTorch分类器 (GPU版本)
            success = self._init_pytorch_classifier()
            
            self.initialized = success
            return success
            
        except Exception as e:
            print(f"初始化分类器失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return False
    
    def _init_sklearn_classifiers(self):
        """已弃用的CPU分类器初始化方法，保留以兼容旧代码"""
        print("警告: 调用了已弃用的CPU分类器初始化方法，使用GPU版本代替")
        return False
    
    def _init_pytorch_classifier(self):
        """初始化PyTorch分类器 (GPU版本) - 优化版本，减少训练轮数，提高特征利用效率"""
        # 设置固定随机种子确保结果可重现
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # 确保历史记录已转换为GPU张量
        if self.history_features_gpu is None or len(self.history_features_gpu) < self.min_samples:
            # 如果GPU历史记录不存在或样本不足，从CPU历史记录转换
            X_cpu = np.array(self.history_features_cpu[:self.args.history_size])
            y_cpu = np.array(self.history_predictions_cpu[:self.args.history_size])
            
            # 转换为二分类标签
            binary_labels = (y_cpu == self.args.target_class).astype(np.float32)
            
            # 转换为GPU张量 - 确保指定了正确的设备
            self.history_features_gpu = torch.FloatTensor(X_cpu).to(DEVICE)
            self.history_predictions_gpu = torch.FloatTensor(binary_labels).to(DEVICE)
            
            # 打印设备信息以验证
            print(f"标签推断数据已加载到: {self.history_features_gpu.device}")
        
        # 确保标签是二分类（0或1）并在正确的设备上
        binary_labels = (self.history_predictions_gpu == self.args.target_class).float()
        
        # 确保神经网络在正确的设备上
        self.inference_classifier_gpu = self.inference_classifier_gpu.to(DEVICE)
        print(f"标签推断分类器在设备: {next(self.inference_classifier_gpu.parameters()).device}")
        
        try:
            # 训练分类器 - 使用数据增强和优化超参数
            print(f"训练标签推断分类器(GPU版)，使用 {len(self.history_features_gpu)} 个样本...")
            self.inference_classifier_gpu.train()
            
            # 启用CUDA基准模式以加速卷积操作
            torch.backends.cudnn.benchmark = True
            
            # 优化版本 - 使用更大批次和更均衡的样本分布
            features = self.history_features_gpu
            labels = binary_labels
            
            # 确保数据在设备上
            if features.device != DEVICE:
                features = features.to(DEVICE)
            if labels.device != DEVICE:
                labels = labels.to(DEVICE)
            
            # 分析数据集的类别平衡情况
            positive_count = labels.sum().item()
            total_count = len(labels)
            negative_count = total_count - positive_count
            pos_ratio = positive_count / total_count if total_count > 0 else 0
            
            print(f"数据集类别分布: 正例(目标类)={positive_count}({pos_ratio:.1%}), 负例(非目标类)={negative_count}({1-pos_ratio:.1%})")
            
            # 计算类别权重以处理不平衡问题
            if pos_ratio < 0.3 or pos_ratio > 0.7:
                print("检测到严重的类别不平衡，应用类别权重...")
                pos_weight = torch.tensor([max(1.0, (1-pos_ratio)/pos_ratio) if pos_ratio > 0 else 1.0]).to(DEVICE)
            else:
                pos_weight = None
            
            # 训练循环 - 保持5轮训练，但优化学习过程
            epochs = 5
            batch_size = 64
            # 使用Adam优化器，带权重衰减 - 降低学习率，提高稳定性
            optimizer = torch.optim.Adam(self.inference_classifier_gpu.parameters(), 
                                        lr=0.005,  # 降低学习率从0.01到0.005
                                        weight_decay=5e-4)  # 增加权重衰减从1e-4到5e-4
            
            # 学习率调度器 - 线性预热后余弦衰减
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, 
                epochs=epochs, steps_per_epoch=math.ceil(len(features)/batch_size),
                pct_start=0.3,  # 30%时间用于预热
                anneal_strategy='cos'
            )
            
            # 使用进度条
            epoch_iterator = tqdm(range(epochs), desc="训练标签推断")
            
            for epoch in epoch_iterator:
                # 手动实现批次处理，避免使用DataLoader
                num_batches = (len(features) + batch_size - 1) // batch_size
                indices = torch.randperm(len(features), device=DEVICE)
                
                total_loss = 0
                correct = 0
                total = 0
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(features))
                    batch_indices = indices[start_idx:end_idx]
                    
                    # 获取批次数据
                    batch_features = features[batch_indices]
                    batch_labels = labels[batch_indices]
                    
                    # 清除梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.inference_classifier_gpu(batch_features).squeeze()
                    
                    # 处理单样本批次情况
                    if len(batch_indices) == 1:
                        outputs = outputs.unsqueeze(0)
                    
                    # 使用BCEWithLogitsLoss，支持类别权重
                    if pos_weight is not None:
                        loss = F.binary_cross_entropy_with_logits(outputs, batch_labels, pos_weight=pos_weight)
                    else:
                        loss = F.binary_cross_entropy_with_logits(outputs, batch_labels)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪避免梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.inference_classifier_gpu.parameters(), 1.0)
                    
                    # 优化器步进
                    optimizer.step()
                    
                    # 更新学习率
                    scheduler.step()
                    
                    # 统计
                    total_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                # 更新进度条
                accuracy = 100 * correct / total
                epoch_iterator.set_postfix(loss=total_loss/num_batches, acc=f"{accuracy:.2f}%")
            
            # 评估模型
            self.inference_classifier_gpu.eval()
            with torch.no_grad():
                outputs = self.inference_classifier_gpu(self.history_features_gpu).squeeze()
                # 处理单样本情况
                if len(self.history_features_gpu) == 1:
                    outputs = outputs.unsqueeze(0)
                # 使用sigmoid获取概率值，然后进行预测
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                binary_labels = (self.history_predictions_gpu == self.args.target_class).float()
                accuracy = (predicted == binary_labels).float().mean().item() * 100
                
                # 计算并打印详细的性能指标
                true_pos = ((predicted == 1) & (binary_labels == 1)).sum().item()
                true_neg = ((predicted == 0) & (binary_labels == 0)).sum().item()
                false_pos = ((predicted == 1) & (binary_labels == 0)).sum().item()
                false_neg = ((predicted == 0) & (binary_labels == 1)).sum().item()
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
                
                print(f"标签推断训练精度(GPU版): {accuracy:.2f}%")
                print(f"推断性能详情: 精确率={precision:.2f}, 召回率={recall:.2f}, 特异性={specificity:.2f}")
                print(f"混淆矩阵: TP={true_pos}, TN={true_neg}, FP={false_pos}, FN={false_neg}")
            
            # 清空缓存以释放GPU内存
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"GPU标签推断训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 清空缓存以释放GPU内存
            torch.cuda.empty_cache()
            
            return False
    
    def infer_labels(self, features, top_model=None, bottom_models=None, raw_data=None):
        """推断输入特征的标签，使用GPU分类器并支持自适应置信度阈值
        
        Args:
            features: 输入特征
            top_model: 顶部模型 (可选)
            bottom_models: 底部模型列表 (可选)
            raw_data: 原始输入数据 (可选)
            
        Returns:
            二分类预测结果和置信度
        """
        # 只使用GPU训练好的推断分类器
        if self.initialized and self.inference_classifier_gpu is not None:
            try:
                # 确保特征是适当的张量类型并在GPU上
                if isinstance(features, torch.Tensor):
                    if features.device != DEVICE:
                        features = features.to(DEVICE)
                else:
                    features = torch.FloatTensor(features).to(DEVICE)
                
                # 确保特征形状正确
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                # 设置自适应批次大小，根据GPU内存使用情况自动调整
                if torch.cuda.is_available():
                    # 获取当前可用GPU内存(MB)
                    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) - torch.cuda.memory_allocated() / (1024 * 1024)
                    # 根据可用内存调整批次大小，每样本估计需要约20MB内存(针对CINIC-10)
                    estimated_mem_per_sample = 20  # MB
                    batch_size = max(16, min(256, int(free_mem / (estimated_mem_per_sample * 1.5))))  # 预留50%内存
                else:
                    batch_size = 64  # CPU默认批次
                
                all_preds = []
                all_confidences = []
                
                # 使用GPU分类器进行推断 - 分批处理以减少内存占用
                self.inference_classifier_gpu.eval()
                with torch.no_grad():
                    for i in range(0, features.size(0), batch_size):
                        end = min(i + batch_size, features.size(0))
                        batch = features[i:end]
                        
                        outputs = self.inference_classifier_gpu(batch).squeeze()
                        # 处理单样本批次
                        if end - i == 1:
                            outputs = outputs.unsqueeze(0)
                            
                        # 添加sigmoid函数获取概率值
                        probs = torch.sigmoid(outputs)
                        
                        # 使用自适应置信度阈值 - 如果设置了自适应阈值选项
                        if hasattr(self.args, 'adaptive_threshold') and self.args.adaptive_threshold:
                            # 根据概率分布动态确定阈值
                            sorted_probs, _ = torch.sort(probs)
                            # 如果分布明显偏向某一端，使用更严格的阈值
                            if sorted_probs[int(len(sorted_probs) * 0.25)] < 0.3:  # 第一四分位数小于0.3
                                threshold = 0.65  # 提高阈值，减少假阳性
                            elif sorted_probs[int(len(sorted_probs) * 0.75)] > 0.7:  # 第三四分位数大于0.7
                                threshold = 0.35  # 降低阈值，减少假阴性
                            else:
                                # 否则使用标准阈值
                                threshold = 0.5
                        else:
                            # 使用固定阈值
                            threshold = self.confidence_threshold if self.confidence_threshold != 0.3 else 0.5
                        
                        # 基于阈值进行决策
                        batch_preds = (probs > threshold).cpu().numpy().astype(np.int32)
                        batch_confidence = probs.cpu().numpy()
                        
                        all_preds.append(batch_preds)
                        all_confidences.append(batch_confidence)
                
                # 合并所有批次的结果
                if len(all_preds) > 1:
                    binary_pred = np.concatenate(all_preds)
                    confidence = np.concatenate(all_confidences)
                else:
                    binary_pred = all_preds[0]
                    confidence = all_confidences[0]
                
                # 应用置信度过滤 - 识别高置信度样本
                high_confidence = np.where(
                    (confidence > 0.8) | (confidence < 0.2)
                )[0]
                
                # 标记低置信度样本比例以便调试
                low_confidence_ratio = 1.0 - (len(high_confidence) / len(binary_pred))
                if low_confidence_ratio > 0.4:  # 如果超过40%的样本置信度较低
                    print(f"警告: 有 {low_confidence_ratio:.1%} 的样本置信度不高，推断可能不准确")
                
                return binary_pred, confidence
                
            except Exception as e:
                print(f"GPU标签推断失败: {str(e)}")
                print(f"无法进行标签推断，返回None")
                import traceback
                traceback.print_exc()  # 打印详细错误信息
                return None, None
        
        # 如果没有初始化或无法推断，返回None
        return None, None

    def get_total_samples(self):
        """获取收集的样本总数"""
        return len(self.history_features_cpu)
        
    def update_class_stats(self, modelC=None, bottom_models=None, force=False):
        """更新类别统计信息"""
        # 检查是否有足够的样本
        if len(self.history_features_cpu) < self.min_samples and not force:
            print(f"样本不足，无法更新类别统计信息 ({len(self.history_features_cpu)}/{self.min_samples})")
            return False
        
        # 初始化或更新分类器
        if not self.initialized or force:
            success = self.initialize_classifier()
            if success:
                print(f"成功初始化标签推断 (样本数: {len(self.history_features_cpu)})")
                return True
            else:
                print(f"标签推断初始化失败")
                return False
        
        return self.initialized

    def embedding_swapping(self):
        """为兼容性添加的方法，在BadVFL中不执行嵌入交换，而是直接使用收集的特征进行标签推断"""
        # 检查是否有足够的样本
        if len(self.history_features_cpu) < self.min_samples:
            print(f"样本不足，无法进行嵌入交换: {len(self.history_features_cpu)}/{self.min_samples}")
            return False
        
        print(f"BadVFL不执行嵌入交换，直接使用收集的 {len(self.history_features_cpu)} 个样本进行标签推断")
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
        # device现在由全局DEVICE变量确定，无需单独传递
        self.device = DEVICE
        self.dataset_name = args.dataset
        self.position = args.position  # 触发器位置
        
        # 为兼容性处理参数名称
        if hasattr(args, 'trigger_size') and not hasattr(args, 'pattern_size'):
            self.pattern_size = int(args.trigger_size)  # 触发器大小
            print(f"触发器初始化: 使用trigger_size ({args.trigger_size}) 作为pattern_size")
        elif hasattr(args, 'pattern_size'):
            self.pattern_size = int(args.pattern_size)  # 触发器大小
        else:
            self.pattern_size = 4  # 默认值
            print(f"触发器初始化: 未找到大小参数，使用默认值 pattern_size = {self.pattern_size}")
        
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
        if self.dataset_name.upper() in ['CINIC10', 'CIFAR10', 'CIFAR100']:
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
        # 确保data在正确的设备上
        device = data.device
        
        # 确保attack_flags是Tensor类型
        if attack_flags is not None and not isinstance(attack_flags, torch.Tensor):
            attack_flags = torch.tensor(attack_flags, dtype=torch.bool, device=device)
        elif attack_flags is not None and attack_flags.device != device:
            attack_flags = attack_flags.to(device)
        
        # 如果没有指定攻击标志，则不注入触发器
        if attack_flags is None or torch.sum(attack_flags) == 0:
            return data
        
        # 克隆数据以避免修改原始数据
        data_copy = data.clone()
        
        # 注入触发器 - 使用更小的触发器尺寸和更自然的混合方式
        if self.trigger_type == 'pixel':
            # 更小的像素块大小，从2x2降低到1x1单个像素
            pixel_block_size = 1
            # 进一步降低触发器强度到原强度的25%
            reduced_intensity = self.intensity * 0.25
            
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 对被攻击样本应用触发器
                    temp_data = data_copy[idx].clone()
                    
                    # 只使用前3个触发点，进一步减少影响区域
                    selected_positions = self.pixel_positions[:3]
                    selected_values = self.pixel_values[:3]
                    
                    for (x, y), (r, g, b) in zip(selected_positions, selected_values):
                        # 使用非常低的透明度，让触发器几乎不可见
                        alpha = 0.3  # 降低透明度
                        temp_data[0, x, y] = (1-alpha) * temp_data[0, x, y] + alpha * r * reduced_intensity
                        temp_data[1, x, y] = (1-alpha) * temp_data[1, x, y] + alpha * g * reduced_intensity
                        temp_data[2, x, y] = (1-alpha) * temp_data[2, x, y] + alpha * b * reduced_intensity
                        
                        # 仅在单个像素周围添加微弱的影响
                        if pixel_block_size > 1:
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    if dx == 0 and dy == 0:  # 跳过中心像素
                                        continue
                                        
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < 32 and 0 <= ny < 32:
                                        # 极快的衰减，几乎不可见
                                        decay = 0.2 * (1.0 - 0.5 * max(abs(dx), abs(dy)))
                                        # 混合原图像和触发器，使用极低的透明度
                                        alpha_edge = 0.2 * decay
                                        temp_data[0, nx, ny] = (1-alpha_edge) * temp_data[0, nx, ny] + alpha_edge * r * reduced_intensity
                                        temp_data[1, nx, ny] = (1-alpha_edge) * temp_data[1, nx, ny] + alpha_edge * g * reduced_intensity
                                        temp_data[2, nx, ny] = (1-alpha_edge) * temp_data[2, nx, ny] + alpha_edge * b * reduced_intensity
                    
                    # 将处理后的数据复制回data_copy
                    data_copy[idx] = temp_data
        
        elif self.trigger_type == 'pattern':
            # 模式触发器 - 减小尺寸并减少对比度
            if not hasattr(self, 'pattern_mask') or self.pattern_mask is None:
                print("错误: 未找到pattern_mask!")
                return data_copy
                
            # 确保pattern_mask在正确的设备上
            pattern_mask = self.pattern_mask
            if pattern_mask.device != device:
                pattern_mask = pattern_mask.to(device)
                self.pattern_mask = pattern_mask
            
            # 使用改进的小模式触发器
            small_pattern_mask = None
            
            # 创建一个更小的触发器模式
            # 这是为了减少触发器的可见性
            # 将触发器大小进一步减小
            small_pattern_size = max(1, int(self.pattern_size / 2))
            
            # 创建更小的pattern_mask
            if not hasattr(self, 'small_pattern_mask'):
                small_pattern = torch.zeros(3, 32, 32, device=device)
                
                # 根据位置确定小触发器的位置
                if self.position == 'dr':  # 右下角
                    x_start, y_start = 32 - small_pattern_size, 32 - small_pattern_size
                elif self.position == 'ul':  # 左上角
                    x_start, y_start = 0, 0 
                elif self.position == 'mid':  # 中间
                    x_start, y_start = 16 - small_pattern_size//2, 16 - small_pattern_size//2
                else:  # 默认右下角
                    x_start, y_start = 32 - small_pattern_size, 32 - small_pattern_size
                
                # 创建简单但有效的模式 - 使用更淡的颜色
                for i in range(small_pattern_size):
                    for j in range(small_pattern_size):
                        if i == j:  # 主对角线，浅红色
                            small_pattern[0, x_start + i, y_start + j] = 0.4  # R - 降低强度
                            small_pattern[1, x_start + i, y_start + j] = 0.1  # G
                            small_pattern[2, x_start + i, y_start + j] = 0.1  # B
                        else:  # 其他位置，浅蓝色
                            small_pattern[0, x_start + i, y_start + j] = 0.1  # R
                            small_pattern[1, x_start + i, y_start + j] = 0.1  # G
                            small_pattern[2, x_start + i, y_start + j] = 0.4  # B - 降低强度
                
                self.small_pattern_mask = small_pattern
            
            small_pattern_mask = self.small_pattern_mask
            
            # 为每个需要攻击的样本应用小触发器
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 创建掩码，指示哪些像素需要修改
                    mask = (small_pattern_mask > 0).float()
                    
                    # 创建临时数据以避免修改原始数据
                    temp_data = data_copy[idx].clone()
                    
                    # 使用极低的强度和低透明度，使触发器几乎不可见
                    reduced_intensity = self.intensity * 0.25  # 极低强度，仅为原来的25%
                    alpha = 0.25  # 非常低的透明度
                    
                    # 使用混合函数，避免完全替换原像素
                    blended = (1 - mask * alpha) * temp_data + small_pattern_mask * reduced_intensity * alpha
                    
                    # 只添加最小的噪声
                    noise = torch.randn_like(temp_data) * 0.005
                    blended = blended + mask * noise
                    
                    # 确保值在有效范围内
                    blended = torch.clamp(blended, 0.0, 1.0)
                    
                    # 将处理后的数据复制回data_copy
                    data_copy[idx] = blended
        
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
            print("警告：标签推断失败，使用原始攻击标志")
            return self.inject_trigger(data, attack_flags)
        
        # 简化标签推断逻辑，确保我们攻击足够多的样本
        # 创建基于推断的攻击标志 - 攻击所有非目标类样本
        inference_attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=data.device)
        
        # 统计推断结果
        target_count = 0
        non_target_count = 0
        
        for i in range(batch_size):
            if inferred_labels[i] == self.target_class:
                target_count += 1
            else:
                non_target_count += 1
                
            # 简化条件: 如果标签不是目标类，我们就攻击
            # 不再使用复杂的条件和置信度判断
            if attack_flags is not None and attack_flags[i] and inferred_labels[i] != self.target_class:
                inference_attack_flags[i] = True
            # 备选条件: 如果预测的标签是我们的源类别(如果定义了)，我们会优先攻击
            elif attack_flags is not None and attack_flags[i] and hasattr(self.args, 'source_class') and inferred_labels[i] == self.args.source_class:
                inference_attack_flags[i] = True
        
        # 计算攻击统计信息
        original_attack_count = attack_flags.sum().item() if attack_flags is not None else 0
        inference_attack_count = inference_attack_flags.sum().item()
        
        # 每10个批次打印一次详细信息
        import random
        if random.random() < 0.1:
            print(f"\n[标签推断攻击] 批次大小={batch_size}")
            print(f"推断结果: 目标类={target_count}, 非目标类={non_target_count}")
            print(f"攻击决策: 原始攻击={original_attack_count}, 基于推断攻击={inference_attack_count}")
        
        # 如果标签推断未能识别任何样本（很少见），退回到原始标志
        if not inference_attack_flags.any() and attack_flags is not None and attack_flags.any():
            print("警告：标签推断没有识别到任何可攻击样本，使用原始攻击标志")
            return self.inject_trigger(data, attack_flags)
        
        # 应用触发器 - 使用超强触发模式确保明显
        return self.inject_trigger(data, inference_attack_flags)

class ResBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # 如果需要下采样，或者输入输出通道数不一致，则使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        # 主路径
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        # 快捷路径
        shortcut = self.shortcut(x)
        
        # 合并主路径和快捷路径 - 使用非in-place操作
        out = out + shortcut
        out = self.relu(out)
        
        return out

class CINIC10BottomModel(nn.Module):
    """CINIC-10底部模型，支持标准BadVFL攻击"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(CINIC10BottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 使用ResNet风格的特征提取器
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # 计算特征图大小
        feature_dim = 128 * (32 // 4) * (32 // 4)  # 经过2次stride=2的下采样
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GroupNorm(8, 512),
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
        layers.append(ResBlock(in_channels, out_channels, stride))
        # 其余块不需要下采样
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
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
        # 确保输入数据在正确的设备上
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
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

class CINIC10TopModel(nn.Module):
    """CINIC-10顶部模型"""
    def __init__(self, input_dim=256, num_classes=10):
        super(CINIC10TopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.gn1 = nn.GroupNorm(8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.gn2 = nn.GroupNorm(8, 256)
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

def download_cinic10(data_root):
    """自动下载CINIC-10数据集"""
    print(f"\n{'='*50}")
    print("CINIC-10数据集未找到，开始自动下载...")
    print(f"{'='*50}")
    
    # CINIC-10数据集的下载链接
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    
    # 创建目标目录
    os.makedirs(data_root, exist_ok=True)
    
    # 下载文件路径
    download_path = os.path.join(data_root, "CINIC-10.tar.gz")
    
    print(f"下载URL: {url}")
    print(f"保存路径: {download_path}")
    print("开始下载... (这可能需要几分钟)")
    
    try:
        # 使用进度条下载
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r下载进度: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='')
            else:
                print(f"\r已下载: {downloaded // (1024*1024)}MB", end='')
        
        urllib.request.urlretrieve(url, download_path, reporthook=show_progress)
        print("\n下载完成!")
        
        # 解压文件
        print("开始解压...")
        with tarfile.open(download_path, 'r:gz') as tar:
            # 获取所有成员
            members = tar.getmembers()
            
            # 使用tqdm显示解压进度
            for member in tqdm(members, desc="解压中"):
                tar.extract(member, data_root)
        
        print("解压完成!")
        
        # 检查解压后的目录结构
        extracted_dir = os.path.join(data_root, "CINIC-10")
        if os.path.exists(extracted_dir):
            # 将CINIC-10子目录的内容移动到data_root
            for item in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, item), os.path.join(data_root, item))
            # 删除空的CINIC-10目录
            os.rmdir(extracted_dir)
            print("目录结构调整完成!")
        
        # 删除压缩文件以节省空间
        os.remove(download_path)
        print("清理完成!")
        
        # 验证下载结果
        train_dir = os.path.join(data_root, 'train')
        test_dir = os.path.join(data_root, 'test')
        valid_dir = os.path.join(data_root, 'valid')
        
        if os.path.exists(train_dir) and (os.path.exists(test_dir) or os.path.exists(valid_dir)):
            print(f"CINIC-10数据集下载并解压成功!")
            print(f"训练集: {train_dir}")
            print(f"测试集: {test_dir if os.path.exists(test_dir) else valid_dir}")
            
            # 统计每个集合的样本数
            try:
                train_count = sum([len(files) for r, d, files in os.walk(train_dir)])
                test_count = sum([len(files) for r, d, files in os.walk(test_dir if os.path.exists(test_dir) else valid_dir)])
                print(f"训练样本数: {train_count}")
                print(f"测试样本数: {test_count}")
            except:
                pass
            
            return True
        else:
            print("解压后目录结构不正确")
            return False
            
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        print("请尝试手动下载CINIC-10数据集")
        print("下载地址: https://datashare.ed.ac.uk/handle/10283/3192")
        return False

def load_dataset(dataset_name, data_dir, batch_size):
    """加载CINIC-10数据集 - 支持自动下载"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    
    # CINIC-10数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
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
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404],
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    
    print("\n2. 检查CINIC-10数据集路径...")
    data_root = data_dir
    
    # 检查数据集是否已存在
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    valid_dir = os.path.join(data_root, 'valid')  # CINIC-10可能使用valid而不是test
    
    dataset_exists = (os.path.exists(train_dir) and 
                     (os.path.exists(test_dir) or os.path.exists(valid_dir)))
    
    if not dataset_exists:
        print(f"数据集未找到: {data_root}")
        print("找不到CINIC-10数据集，请确认数据集路径")
        
        # 尝试自动下载
        print("尝试自动下载CINIC-10数据集...")
        download_success = download_cinic10(data_root)
        
        if not download_success:
            print("自动下载失败")
            sys.exit(1)
        
        # 重新检查数据集
        dataset_exists = (os.path.exists(train_dir) and 
                         (os.path.exists(test_dir) or os.path.exists(valid_dir)))
        
        if not dataset_exists:
            print("下载后仍无法找到数据集")
            sys.exit(1)
    else:
        print(f"找到已有数据集: {data_root}")
    
    print("\n3. 加载CINIC-10数据集...")
    
    # 使用ImageFolder加载数据集
    try:
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform_train
        )
        
        # 检查测试集路径
        if os.path.exists(test_dir):
            test_dataset = datasets.ImageFolder(
                root=test_dir,
                transform=transform_test
            )
        elif os.path.exists(valid_dir):
            test_dataset = datasets.ImageFolder(
                root=valid_dir,
                transform=transform_test
            )
        else:
            raise FileNotFoundError("找不到测试集或验证集目录")
        
        print(f"CINIC-10数据集加载成功!")
        print(f"类别数量: {len(train_dataset.classes)}")
        print(f"类别列表: {train_dataset.classes}")
        
        # 验证数据集
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("数据集为空")
        
        if len(train_dataset.classes) != 10:
            print(f"警告: 期望10个类别，实际找到{len(train_dataset.classes)}个类别")
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("请检查数据集路径和格式")
        sys.exit(1)
    
    print("\n4. 创建数据加载器...")
    
    # 数据加载器配置
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"数据集统计信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 验证数据加载器
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

def prepare_backdoor_data(data, target, args=None):
    """准备后门数据，注入后门触发器 - 确保设备一致性"""
    global DEVICE
    batch_size = data.size(0)
    
    # 确保数据在正确的设备上
    data_device = data.device
    
    # 首先获取所有非目标类样本的索引
    non_target_indices = [i for i in range(batch_size) if target[i] != args.target_class]
    
    # 如果没有非目标类样本，无法进行攻击
    if len(non_target_indices) == 0:
        # 创建空的攻击标志
        attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=data_device)
        print("警告: 当前批次没有非目标类样本可攻击!")
        return data.clone(), target.clone(), attack_flags
    
    # 大幅降低攻击比例，使其保持在10%-20%之间的随机值
    # 这样可以避免攻击过多样本导致ASR计算偏差，更符合实际攻击场景
    attack_ratio = random.uniform(0.1, 0.2)
    max_attack_samples = int(batch_size * attack_ratio)
    max_possible = min(len(non_target_indices), max_attack_samples)
    attack_portion = max(1, max_possible)  # 确保至少有1个样本被攻击
    
    # 设置攻击标志 - 只攻击非目标类样本
    attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=data_device)
    
    # 随机选择attack_portion个非目标类样本
    if len(non_target_indices) > attack_portion:
        selected_indices = random.sample(non_target_indices, attack_portion)
    else:
        selected_indices = non_target_indices
        
    # 设置选中样本的攻击标志
    for idx in selected_indices:
        attack_flags[idx] = True
    
    # 修改标签为目标类别
    bkd_target = target.clone()
    bkd_target[attack_flags] = args.target_class
    
    # 生成克隆的数据，后面将在此基础上应用触发器
    bkd_data = data.clone()
    
    # 打印调试信息，用于排查ASR计算问题
    attack_count = attack_flags.sum().item()
    if attack_count == 0:
        print(f"警告: 准备了 {batch_size} 个样本但没有标记任何攻击样本!")
    
    return bkd_data, bkd_target, attack_flags

def save_checkpoint(modelC, bottom_models, epoch, clean_acc, asr=None, inference_acc=None):
    """保存模型检查点"""
    os.makedirs(ARGS.checkpoint_dir, exist_ok=True)
    
    # 区分DPSGD模式和其他模式，用于文件名
    temp = 'DPSGD' if ARGS.defense_type.upper() == 'DPSGD' else 'ALL'
    label_knowledge = "True" if ARGS.has_label_knowledge else "False"
    
    if asr is None:
        model_name = f"{ARGS.dataset}_Clean_{temp}_{label_knowledge}_{ARGS.party_num}"
    else:
        model_name = f"{ARGS.dataset}_BadVFL_WithInference_{temp}_{label_knowledge}_{ARGS.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(ARGS.checkpoint_dir, model_file_name)
    
    checkpoint = {
        'model_bottom': {f'bottom_model_{i}': model.state_dict() for i, model in enumerate(bottom_models)},
        'model_top': modelC.state_dict(),
        'epoch': epoch,
        'clean_acc': clean_acc,
        'asr': asr,
        'inference_acc': inference_acc,
        'attack_type': 'BadVFL_WithInference',
        'trigger_magnitude': ARGS.trigger_intensity,
        'trigger_size': ARGS.trigger_size,
        'poison_budget': ARGS.poison_budget,
        'inference_weight': ARGS.inference_weight,
        'args': ARGS
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存模型到 {model_save_path}')

def train_epoch(modelC, bottom_models, optimizers, optimizerC, train_loader, epoch, args, label_inference_module=None, defense_hooks=None):
    """训练一个轮次，包括BadVFL后门注入和标签推断"""
    # 动态设置DEVICE和ARGS
    global DEVICE
    global ARGS
    
    is_dpsgd = ARGS.defense_type.upper() == 'DPSGD'

    if defense_hooks is None:
        defense_hooks = SimpleNamespace(forward=None, instance=None)
    
    # Setup models and optimizers based on defense
    if is_dpsgd:
        vfl_system = modelC
        optimizer = optimizers
        vfl_system.train()
        actual_bottom_models = [m._module for m in vfl_system.bottom_models]
        actual_top_model = vfl_system.top_model._module
    else:
        modelC.train()
        for model in bottom_models:
            model.train()
        actual_bottom_models = bottom_models
        actual_top_model = modelC

    # 启用性能优化
    torch.backends.cudnn.benchmark = True
    
    # 初始化统计变量
    total_loss = 0.0
    correct = 0
    backdoor_correct = 0
    total = 0
    backdoor_samples = 0
    
    # 定义梯度累积步数 - 默认为1(不进行梯度累积)
    grad_accumulation_steps = 1
    
    # 初始化CUDA流变量
    compute_stream = None
    data_stream = None
    
    # 获取恶意模型
    adversary_model = actual_bottom_models[ARGS.bkd_adversary]
    
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # 确定是否应用后门攻击
    apply_backdoor = epoch >= ARGS.Ebkd
    
    has_inference = hasattr(adversary_model, 'label_inference') and label_inference_module is not None
    if has_inference:
        # print(f"标签推断模块状态: {'已初始化' if label_inference_module.initialized else '未初始化'}")
        history_len = len(label_inference_module.history_features_cpu) if hasattr(label_inference_module, 'history_features_cpu') else 0
        # print(f"当前收集到的样本数: {history_len}")
    
    # 记录批次数
    batch_count = 0
    # 增加早期阶段的收集批次数量
    warmup_batches = min(100 if epoch < ARGS.Ebkd else 50, len(train_loader) // 10)
    
    # 修改后门损失的权重 - 使用更低的初始权重和更慢的增长速度
    if epoch >= ARGS.Ebkd:
        # 初始权重大幅降低为原始权重的30%
        base_weight = ARGS.backdoor_weight * 0.3
        
        # 使用更缓和的增长曲线
        # 起始阶段（前10个epoch)增长非常缓慢
        if epoch - ARGS.Ebkd < 10:
            epoch_factor = 0.01 * (epoch - ARGS.Ebkd)  # 每个epoch只增加1%
        else:
            # 之后增长速度略微加快，但仍然缓慢
            epoch_factor = 0.1 + 0.005 * (epoch - ARGS.Ebkd - 10)  # 基础增加10%后，每个epoch增加0.5%
        
        # 将最高增幅限制在1.5倍
        backdoor_weight_multiplier = min(1.5, 1.0 + epoch_factor)
        backdoor_weight = base_weight * backdoor_weight_multiplier
        
        # 计算当前权重与干净损失的比例
        weight_ratio = backdoor_weight / 1.0
        
        # 打印调试信息
        initial_weight = ARGS.backdoor_weight * 0.3  # 初始权重
        max_weight = initial_weight * 1.5  # 最大权重
        # print(f"当前后门损失权重: {backdoor_weight:.2f} (初始:{initial_weight:.2f}, 最大:{max_weight:.2f}, 比例:{weight_ratio:.2f})")
    else:
        backdoor_weight_multiplier = 0.0
        backdoor_weight = 0.0
    
    # if apply_backdoor:
    #     print(f"当前后门损失权重: {backdoor_weight:.2f}")
    # else:
    #     print(f"当前为正常训练阶段，尚未开始后门攻击 (开始轮次: {ARGS.Ebkd})")
    #     if has_inference:
    #         print(f"标签推断模块状态: {'已初始化' if label_inference_module.initialized else '未初始化'}")
    #         history_len = len(label_inference_module.history_features_cpu) if hasattr(label_inference_module, 'history_features_cpu') else 0
    #         print(f"当前收集到的样本数: {history_len}")
    
    # 不再使用预取策略，直接使用简单的数据加载
    prefetch_data = None
    prefetch_target = None
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc=f"训练 (Epoch {epoch})")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 增加批次计数
        batch_count += 1
        
        # 确保数据在GPU上 - 显式检查和传输
        if data.device != DEVICE:
            data = data.to(DEVICE, non_blocking=True)
        if target.device != DEVICE:
            target = target.to(DEVICE, non_blocking=True)
            
        # 确保数据是浮点类型
        if data.dtype != torch.float32:
            data = data.float()
            
        # 记录数据设备位置(测试用)
        # if batch_idx == 0:
            # print(f"批次数据设备: {data.device}, 标签设备: {target.device}")
            
        # 清空梯度
        if is_dpsgd:
            optimizer.zero_grad()
        else:
            for opt in optimizers:
                opt.zero_grad()
            optimizerC.zero_grad()
        
        total += len(data)
        
        # 如果使用梯度累积，只有在需要更新时才清零梯度
        if batch_count % grad_accumulation_steps == 1:
            if is_dpsgd:
                optimizer.zero_grad()
            else:
                for opt in optimizers:
                    opt.zero_grad()
                optimizerC.zero_grad()
        
        # Forward pass for clean data
        if is_dpsgd:
            output_clean = vfl_system(data)
        else:
            bottom_outputs_clean = []
            for i, model in enumerate(bottom_models):
                output = model(data)
                bottom_outputs_clean.append(output)
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            if defense_hooks.forward:
                combined_output_clean = defense_hooks.forward(combined_output_clean)
            output_clean = modelC(combined_output_clean)
            
        loss_clean = criterion(output_clean, target)
        
        # 默认使用干净损失
        loss = loss_clean
        
        # 前向传播 - 只有在开始后门攻击后才注入后门触发器
        if apply_backdoor:
            # 准备后门数据 - 确保传递args参数
            bkd_data, bkd_target, attack_flags = prepare_backdoor_data(data, target, ARGS)
            current_backdoor_samples = attack_flags.sum().item()
            backdoor_samples += current_backdoor_samples
            
            # 只有当有样本被攻击时才进行后门训练
            if current_backdoor_samples > 0:
                # 使用基于标签推断的智能触发器注入策略
                if adversary_model.badvfl_trigger is not None:
                    if has_inference and label_inference_module.initialized:
                        # 使用标签推断指导的触发器注入
                        bkd_data = adversary_model.badvfl_trigger.inject_trigger_with_inference(
                            bkd_data, attack_flags, data, actual_top_model, actual_bottom_models
                        )
                    else:
                        # 标签推断未初始化，使用传统方法
                        bkd_data = adversary_model.badvfl_trigger.inject_trigger(bkd_data, attack_flags)
                    
                # 确保后门数据在GPU上
                if bkd_data.device != DEVICE:
                    bkd_data = bkd_data.to(DEVICE)
                if bkd_target.device != DEVICE:
                    bkd_target = bkd_target.to(DEVICE)
                
                # Backward pass for backdoor data
                if is_dpsgd:
                    output_bkd = vfl_system(bkd_data)
                else:
                    bottom_outputs_bkd = []
                    for i, model in enumerate(bottom_models):
                        # 所有模型处理相同的数据 - 触发器已经被注入
                        output = model(bkd_data)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    if defense_hooks.forward:
                        combined_output_bkd = defense_hooks.forward(combined_output_bkd)
                    output_bkd = modelC(combined_output_bkd)
                
                loss_bkd = criterion(output_bkd, bkd_target)
                
                # 组合损失 - 使用降低后的后门损失权重
                # 使用动态权重平衡干净损失和后门损失
                clean_weight = 1.0  # 增加干净损失的权重
                loss = (clean_weight * loss_clean + backdoor_weight * loss_bkd) / (clean_weight + backdoor_weight)
                
                # 计算后门准确率 - 只考虑被攻击的样本
                pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                
                # 获取被攻击样本的预测和目标并比较
                # 改进版：使用矩阵运算而非循环，提高效率
                attack_success = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags]))
                current_backdoor_correct = attack_success.sum().item()
                backdoor_correct += current_backdoor_correct
        
        # 反向传播
        loss.backward()
        
        # 恶意方收集梯度信息用于标签推断 - 只在预训练阶段更积极收集
        if has_inference and epoch < ARGS.Ebkd:
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                # 更新标签推断历史 - 直接使用原始图像数据
                original_data = saved_data.view(saved_data.size(0), -1)
                
                # 只有当历史样本未达到上限时才添加新样本
                if len(label_inference_module.history_features_cpu) < ARGS.history_size:
                    # 修复：使用真实标签而不是梯度数据训练标签推断模块
                    label_inference_module.update_with_batch(original_data, target)
        
        # 梯度裁剪
        if is_dpsgd:
            # Opacus handles gradient clipping as part of the PrivacyEngine
            pass
        else:
            torch.nn.utils.clip_grad_norm_(modelC.parameters(), 1.0)
            for model in bottom_models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新参数
        if is_dpsgd:
            optimizer.step()
        else:
            for opt in optimizers:
                opt.step()
            optimizerC.step()
        
        # 计算准确率
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        batch_correct = pred_clean.eq(target.view_as(pred_clean)).sum().item()
        correct += batch_correct
        
        # 累积损失 - 与梯度累积不同，这里我们乘回来以获得真实损失值
        total_loss += loss.item() * grad_accumulation_steps
        
        # 确保当前批次计算完成后再处理下一批
        if compute_stream and data_stream:
            torch.cuda.current_stream().wait_stream(compute_stream)
            torch.cuda.current_stream().wait_stream(data_stream)
        
        # 打印进度
        if batch_idx % ARGS.log_interval == 0:
            progress = 100. * batch_idx / len(train_loader)
            # print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({progress:.0f}%)]\tLoss: {loss.item() * grad_accumulation_steps:.6f}')
            # 添加GPU内存使用监控
            # if torch.cuda.is_available():
                # print(get_gpu_memory_usage())
    
    # 所有CUDA操作完成后同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    # 计算攻击成功率 - 添加安全检查
    attack_success_rate = 0.0
    if apply_backdoor and backdoor_samples > 0:
        # 安全检查
        if backdoor_correct > backdoor_samples:
            print(f"警告: 训练中backdoor_correct ({backdoor_correct}) > backdoor_samples ({backdoor_samples})")
            backdoor_correct = backdoor_samples  # 安全限制
            
        # 正确计算攻击成功率，确保不会溢出
        attack_success_rate = min(100.0, 100.0 * backdoor_correct / backdoor_samples)
        
        # 如果后期ASR异常高，降低后门权重以防止对主任务的过度干扰
        if epoch > ARGS.Ebkd + 10 and attack_success_rate > 95:
            # 对于后期训练，如果ASR已经很高，轻微降低后门权重
            backdoor_weight = backdoor_weight * 0.9
            # print(f"检测到高ASR，降低后门权重至: {backdoor_weight:.2f}")
    
    # 测试标签推断性能
    inference_accuracy = 0.0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # 创建测试子集
        test_subset_loader = torch.utils.data.Subset(
            train_loader.dataset, 
            indices=range(min(500, len(train_loader.dataset)))
        )
        test_subset_loader = torch.utils.data.DataLoader(
            test_subset_loader, 
            batch_size=ARGS.batch_size, 
            shuffle=False,
            num_workers=0  # 禁用多进程以避免CUDA错误
        )
        
        # 仅用于评估推断性能，不需要更新模型
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_subset_loader:
                # 确保数据在正确的设备上
                if data.device != DEVICE:
                    data = data.to(DEVICE)
                if target.device != DEVICE:
                    target = target.to(DEVICE)
                
                # 使用标签推断预测 - 直接使用原始图像数据
                original_data = data.view(data.size(0), -1)
                inferred_labels, inferred_confidence = label_inference_module.infer_labels(original_data)
                
                if inferred_labels is not None:
                    # 计算推断准确率 - 使用二分类计算方式
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        is_target_class = (true == ARGS.target_class)
                        
                        # 判断预测是否正确
                        if (is_target_class and pred == 1) or (not is_target_class and pred == 0):
                            correct_predictions += 1
                        
                        total_samples += 1
        
            # 计算推断准确率
            if total_samples > 0:
                inference_accuracy = 100.0 * correct_predictions / total_samples
    
    # 清理缓存以释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None, defense_hooks=None):
    """测试模型性能，包括干净准确率和后门攻击成功率"""
    is_dpsgd = ARGS.defense_type.upper() == 'DPSGD'

    if defense_hooks is None:
        defense_hooks = SimpleNamespace(forward=None, instance=None)

    # Setup models for evaluation
    if is_dpsgd:
        vfl_system = modelC
        vfl_system.eval()
        actual_bottom_models = [m._module for m in vfl_system.bottom_models]
        actual_top_model = vfl_system.top_model._module
    else:
        modelC.eval()
        for model in bottom_models:
            model.eval()
        actual_bottom_models = bottom_models
        actual_top_model = modelC
    
    # 启用cudnn基准测试提高性能
    torch.backends.cudnn.benchmark = True
    
    # 定义apply_backdoor变量，与train_epoch函数中保持一致
    apply_backdoor = epoch >= ARGS.Ebkd
    
    # 初始化统计变量
    test_loss = 0.0
    correct = 0
    backdoor_correct = 0
    backdoor_samples = 0
    total = 0
    
    # 初始化非目标类样本计数器 - 用于调试ASR计算
    non_target_total = 0
    non_target_batches = 0  # 包含非目标类样本的批次数量
    
    # 获取恶意模型和标签推断模块
    adversary_model = actual_bottom_models[ARGS.bkd_adversary]
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(test_loader, desc="测试" + (" (带后门)" if is_backdoor else ""))
    
    # 极大减小测试中的攻击样本比例，只攻击5%的非目标类样本
    MAX_TEST_ATTACK_RATIO = 0.05
    
    # 设置最大总攻击样本数，避免测试集中使用太多样本
    MAX_TOTAL_ATTACK_SAMPLES = 1000
    current_total_attack_samples = 0
    
    # 随机选择批次，避免偏向特定位置的批次
    batch_indices = list(range(len(test_loader)))
    random.shuffle(batch_indices)
    selected_batch_indices = set(batch_indices[:min(100, len(batch_indices))])
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            # 确保数据在正确的设备上
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)
            total += batch_size
            
            # 记录非目标类样本数量 - 用于调试
            current_non_target = (target != ARGS.target_class).sum().item()
            non_target_total += current_non_target
            if current_non_target > 0:
                non_target_batches += 1
            
            # 准备干净数据的预测
            if is_dpsgd:
                output_clean = vfl_system(data)
            else:
                bottom_outputs_clean = []
                for model in bottom_models:
                    output = model(data)
                    bottom_outputs_clean.append(output)
                
                combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
                if defense_hooks.forward:
                    combined_output_clean = defense_hooks.forward(combined_output_clean)
                output_clean = modelC(combined_output_clean)
            
            # 计算损失和准确率
            test_loss += criterion(output_clean, target).item()
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 如果需要测试后门攻击
            if is_backdoor and apply_backdoor:
                # 检查批次中是否有非目标类样本
                non_target_mask = (target != ARGS.target_class)
                non_target_count = non_target_mask.sum().item()
                
                # 如果超出了最大总攻击样本数或没有非目标类样本，则跳过
                if current_total_attack_samples >= MAX_TOTAL_ATTACK_SAMPLES or non_target_count == 0:
                    continue
                
                # 只在随机选择的批次中执行攻击以减少总攻击样本数
                if batch_idx not in selected_batch_indices:
                    continue
                
                # 设置一个很小的攻击样本比例
                target_attack_ratio = MAX_TEST_ATTACK_RATIO
                max_attack_samples = int(non_target_count * target_attack_ratio)
                max_attack_samples = max(1, max_attack_samples)  # 确保至少攻击1个样本
                
                # 限制当前批次的攻击样本数，使总数不超过MAX_TOTAL_ATTACK_SAMPLES
                remaining_capacity = MAX_TOTAL_ATTACK_SAMPLES - current_total_attack_samples
                max_attack_samples = min(max_attack_samples, remaining_capacity)
                
                if max_attack_samples <= 0:
                    continue
                
                # 随机选择部分非目标类样本进行攻击
                non_target_indices = non_target_mask.nonzero(as_tuple=True)[0]
                
                if len(non_target_indices) > max_attack_samples:
                    # 只随机选择一小部分样本
                    selected_indices = non_target_indices[torch.randperm(len(non_target_indices))[:max_attack_samples]]
                    attack_flags = torch.zeros_like(non_target_mask)
                    attack_flags[selected_indices] = True
                else:
                    # 如果样本太少，不一定全部选择
                    num_to_select = min(len(non_target_indices), max_attack_samples)
                    selected_indices = non_target_indices[torch.randperm(len(non_target_indices))[:num_to_select]]
                    attack_flags = torch.zeros_like(non_target_mask)
                    attack_flags[selected_indices] = True
                
                # 更新总攻击样本计数
                current_attack_count = attack_flags.sum().item()
                current_total_attack_samples += current_attack_count
                
                # 记录攻击样本数
                backdoor_samples += current_attack_count
                
                # 准备后门数据
                bkd_data = data.clone()
                bkd_target = target.clone()
                
                # 修改攻击样本的标签为目标类
                bkd_target[attack_flags] = ARGS.target_class
                
                # 注入触发器
                if hasattr(adversary_model, 'badvfl_trigger') and adversary_model.badvfl_trigger is not None:
                    # 使用基于标签推断的智能触发器注入策略
                    if (hasattr(adversary_model, 'label_inference') and 
                        adversary_model.label_inference is not None and 
                        adversary_model.label_inference.initialized):
                        # 使用标签推断指导的触发器注入
                        bkd_data = adversary_model.badvfl_trigger.inject_trigger_with_inference(
                            bkd_data, attack_flags, data, actual_top_model, actual_bottom_models
                        )
                    else:
                        # 标签推断未初始化，使用传统方法
                        bkd_data = adversary_model.badvfl_trigger.inject_trigger(bkd_data, attack_flags)
                
                # 前向传播 - 对包含注入触发器的样本
                if is_dpsgd:
                    output_bkd = vfl_system(bkd_data)
                else:
                    bottom_outputs_bkd = []
                    for model in bottom_models:
                        output = model(bkd_data)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    if defense_hooks.forward:
                        combined_output_bkd = defense_hooks.forward(combined_output_bkd)
                    output_bkd = modelC(combined_output_bkd)
                
                # 只考虑被攻击样本的预测
                pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                
                # 计算攻击成功的样本数 - 只考虑被修改标签的样本
                attack_success = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags]))
                backdoor_correct += attack_success.sum().item()
    
    # 计算平均损失和准确率
    test_loss /= total
    clean_acc = 100. * correct / total
    
    # 计算ASR（如果有后门样本）
    asr = 0.0
    if backdoor_samples > 0:
        asr = 100. * backdoor_correct / backdoor_samples
    
    # 打印调试信息
    if is_backdoor and apply_backdoor:
        print(f"\nASR调试信息:")
        print(f"总样本数: {total}, 非目标类样本总数: {non_target_total}")
        print(f"包含非目标类样本的批次: {non_target_batches}/{len(test_loader)}")
        print(f"攻击样本数: {backdoor_samples}, 攻击成功数: {backdoor_correct}")
        print(f"最大攻击比例: {MAX_TEST_ATTACK_RATIO*100:.1f}%, 最大攻击样本数: {MAX_TOTAL_ATTACK_SAMPLES}")
    
    # 打印结果
    print(f'\n测试结果:')
    print(f'损失: {test_loss:.4f}, 准确率: {clean_acc:.2f}%')
    if is_backdoor and apply_backdoor:
        print(f'ASR: {asr:.2f}% (正确: {backdoor_correct}/{backdoor_samples})')
    
    return {
        'loss': test_loss,
        'clean_acc': clean_acc,
        'asr': asr
    }

def collect_inference_data(modelC, bottom_models, train_loader, args):
    """收集标签推断数据，使用BadVFL的标签推断方法 - 优化版本"""
    print("\n启动BadVFL标签推断过程...")

    modelC.train()
    for model in bottom_models:
        model.train()

    adversary_model = bottom_models[ARGS.bkd_adversary]
    label_inference_module = adversary_model.label_inference

    if not label_inference_module:
        print("错误: 恶意模型没有标签推断模块")
        return False

    # 采集目标
    min_target = 30
    min_non_target = 300
    max_total = 1000

    class_counts = {'target': 0, 'non_target': 0}
    total_collected = 0

    optimizers = [optim.SGD(model.parameters(), lr=ARGS.lr) for model in bottom_models]
    optimizerC = optim.SGD(modelC.parameters(), lr=ARGS.lr)

    print(f"将采集至少 {min_target} 个目标类、{min_non_target} 个非目标类，总计不低于 {max_total} 个样本用于标签推断")

    batch_progress = tqdm(total=max_total // ARGS.batch_size + 10, desc="收集梯度数据")

    for batch_idx, (data, target) in enumerate(train_loader):
        if total_collected >= max_total and \
           class_counts['target'] >= min_target and \
           class_counts['non_target'] >= min_non_target:
            print(f"\n已采集到足够样本，目标类: {class_counts['target']}，非目标类: {class_counts['non_target']}")
            break

        data, target = data.to(DEVICE), target.to(DEVICE)

        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()

        bottom_outputs = []
        for i, model in enumerate(bottom_models):
            if i == ARGS.bkd_adversary:
                data.requires_grad_(True)
            output = model(data)
            bottom_outputs.append(output)

        combined_output = torch.cat(bottom_outputs, dim=1)
        output = modelC(combined_output)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()

        saved_data, saved_grad = adversary_model.get_saved_data()
        if saved_data is not None and saved_grad is not None:
            original_data = saved_data.view(saved_data.size(0), -1)
            # 修复：使用真实标签而不是梯度数据训练标签推断模块
            samples_added = label_inference_module.update_with_batch(original_data, target)
            # 统计类别
            target_np = target.detach().cpu().numpy()
            class_counts['target'] += np.sum(target_np == ARGS.target_class)
            class_counts['non_target'] += np.sum(target_np != ARGS.target_class)
            total_collected += len(target_np)
            batch_progress.update(1)
            batch_progress.set_postfix(samples=total_collected, target=class_counts['target'], non_target=class_counts['non_target'])

    batch_progress.close()
    print(f"最终采集到 样本总数: {total_collected}, 目标类: {class_counts['target']}, 非目标类: {class_counts['non_target']}")

    # 初始化推断器
    if len(label_inference_module.history_features_cpu) >= label_inference_module.min_samples:
        if label_inference_module.initialize_classifier():
            if adversary_model.badvfl_trigger:
                adversary_model.badvfl_trigger.update_inference_stats()
                return True

    return len(label_inference_module.history_features_cpu) >= label_inference_module.min_samples

def warmup_gpu():
    """预热GPU，确保正确初始化"""
    if not torch.cuda.is_available():
        return
    
    print("\nGPU预热中...")
    
    # 打印GPU设备信息
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 创建大规模随机张量并进行各种操作以预热CUDA核心
    x = torch.randn(1000, 1000, device=DEVICE)
    y = torch.randn(1000, 1000, device=DEVICE)
    
    # 进行多种矩阵操作，确保各类CUDA核心被激活
    torch.cuda.synchronize()  # 确保操作开始前同步
    start_time = time.time()
    
    print("执行矩阵操作...")
    for _ in range(10):
        z1 = torch.matmul(x, y)  # 矩阵乘法
        z2 = torch.relu(z1)      # 激活函数
        z3 = torch.sigmoid(z1)   # 另一个激活函数
        z4 = z2 + z3             # 元素加法
        z5 = z4 * 0.5            # 元素乘法
        z6 = torch.tanh(z5)      # tanh激活
        z7 = torch.mean(z6, dim=1)  # 归约操作
        
        # 防止编译器优化掉未使用的张量
        tmp = z7.sum().item()
    
    torch.cuda.synchronize()  # 确保操作结束后同步
    
    # 预热卷积操作
    print("执行卷积操作...")
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(DEVICE)
    batch = torch.randn(16, 3, 32, 32, device=DEVICE)
    
    for _ in range(5):
        out = conv(batch)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        tmp = out.sum().item()
    
    # 预热优化器操作
    print("执行优化器操作...")
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(DEVICE)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    input_data = torch.randn(32, 1000, device=DEVICE)
    target = torch.randint(0, 10, (32,), device=DEVICE)
    
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    
    # 强制同步并清空缓存
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # 测量总时间
    elapsed_time = time.time() - start_time
    print(f"GPU预热完成，耗时: {elapsed_time:.2f}秒")
    
    # 报告GPU内存状态
    print(get_gpu_memory_usage())

class VFLSystem(nn.Module):
    """VFL系统的包装类，用于DPSGD"""
    def __init__(self, bottom_models, top_model):
        super().__init__()
        # 将每个底部模型包装为GradSampleModule
        self.bottom_models = nn.ModuleList([
            GradSampleModule(model) for model in bottom_models
        ])
        # 将顶部模型包装为GradSampleModule
        self.top_model = GradSampleModule(top_model)
    
    def forward(self, x):
        # 前向传播
        bottom_outputs = []
        for model in self.bottom_models:
            output = model(x)
            bottom_outputs.append(output)
        
        combined_output = torch.cat(bottom_outputs, dim=1)
        output = self.top_model(combined_output)
        return output

def main():
    """主函数 - 重构版本，确保正确使用GPU"""
    global ARGS
    global DEVICE
    
    print("\n======== 开始初始化 ========")
    # 检查CUDA可用性 
    cuda_available = check_cuda()
    
    if not cuda_available:
        print("错误: 无可用GPU，程序将退出")
        return
    
    # 设置随机种子
    setup_seed(ARGS.seed)
    
    # GPU和DEVICE的设置已在文件顶部通过 CUDA_VISIBLE_DEVICES 完成，此处无需任何操作
    
    
    if ARGS.lr < 0.002:
        print(f"将学习率从 {ARGS.lr} 增加到 0.002 以加快训练")
        ARGS.lr = 0.002
    
    # 3. 打印训练参数
    print("\n" + "="*50)
    print("训练参数:")
    print(f"物理GPU ID: {ARGS.gpu}")
    print(f"PyTorch设备: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"批处理大小: {ARGS.batch_size}")
    print(f"学习率: {ARGS.lr}")
    print(f"轮次: {ARGS.epochs}")
    print(f"后门注入开始轮次: {ARGS.Ebkd}")
    print(f"后门损失权重: {ARGS.backdoor_weight}")
    print("="*50 + "\n")
    
    # 4. 确保参数一致性
    if hasattr(ARGS, 'trigger_size') and hasattr(ARGS, 'pattern_size'):
        if ARGS.trigger_size != ARGS.pattern_size:
            print(f"注意: 统一触发器大小 {ARGS.trigger_size} -> {ARGS.pattern_size}")
            ARGS.pattern_size = ARGS.trigger_size
    
    # 5. 预热GPU
    print("\n预热GPU...")
    warmup_gpu()
    
    # 6. 创建检查点目录
    os.makedirs(ARGS.checkpoint_dir, exist_ok=True)
    
    # 7. 加载数据集
    print("\n加载数据集...")
    train_loader, test_loader = load_dataset(ARGS.dataset, ARGS.data_dir, ARGS.batch_size)
    
    # 8. 创建模型
    print("\n创建模型...")
    bottom_models, modelC = create_models()
    
    # 简化后的验证
    if torch.cuda.is_available():
        for i, model in enumerate(bottom_models):
            device_str = next(model.parameters()).device
            print(f"底部模型 {i}: {device_str}")
        
        device_str = next(modelC.parameters()).device
        print(f"顶部模型: {device_str}")
    
    # 9. 创建优化器
    print("\n创建优化器...")
    optimizers = [optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum) for model in bottom_models]
    optimizerC = optim.SGD(modelC.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    
    # 10. 收集标签推断数据
    print("\n收集标签推断数据...")
    adversary_model = bottom_models[ARGS.bkd_adversary]
    
    # 确保触发器已设置
    if not hasattr(adversary_model, 'badvfl_trigger') or adversary_model.badvfl_trigger is None:
        print("设置BadVFL触发器...")
        badvfl_trigger = BadVFLTrigger(ARGS)
        adversary_model.set_badvfl_trigger(badvfl_trigger)
    
    # 收集标签推断数据
    collect_inference_data(modelC, bottom_models, train_loader, ARGS)
    
    # =================== Defense Setup ===================
    defense_hooks = SimpleNamespace(forward=None, instance=None)
    defense_type = ARGS.defense_type.upper()
    
    # Default optimizers for non-dpsgd case
    optimizer = None
    schedulers = []

    if defense_type != 'NONE':
        print(f"\n{'='*20} Defense Setup: {ARGS.defense_type} {'='*20}")
        
        if ARGS.defense_type.upper() == "DPSGD":
            vfl_system = VFLSystem(bottom_models, modelC).to(DEVICE)
            optimizer = optim.SGD(vfl_system.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=5e-4)
    
            _, optimizer, defense_hooks = build_defense(
                model=vfl_system,
                optimizer=optimizer,
                defense_type=ARGS.defense_type,
                batch_size=ARGS.batch_size,
                noise_multiplier=ARGS.dpsgd_noise_multiplier,
                max_grad_norm=ARGS.dpsgd_max_grad_norm,
                sample_size=len(train_loader.dataset)
            )
            
            optimizerC = None # Not needed for DPSGD
            schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ARGS.epochs))
            defense_hooks.instance = vfl_system
            print("DPSGD setup complete with GradSampleModule wrapping")

        else:
            if ARGS.defense_type.upper() in ["MP", "ANP"]:
                all_models = bottom_models + [modelC]
                all_optimizers = optimizers + [optimizerC]
                for i, (model, opt) in enumerate(zip(all_models, all_optimizers)):
                    print(f"Applying {ARGS.defense_type} to model {i}...")
                    params = {}
                    if ARGS.defense_type.upper() == "MP":
                        params = {'pruning_amount': ARGS.mp_pruning_amount}
                    elif ARGS.defense_type.upper() == "ANP":
                        params = {'sigma': ARGS.anp_sigma}
                    
                    _, _, hooks = build_defense(model, opt, defense_type=ARGS.defense_type, **params)
                    # Note: These defenses might not have specific hooks to apply during forward pass
            
            elif ARGS.defense_type.upper() in ["BDT", "VFLIP", "ISO"]:
                input_dim = modelC.fc1.in_features
                params = {'device': DEVICE, 'input_dim': input_dim}
                
                if ARGS.defense_type.upper() == "BDT":
                    params['prune_ratio'] = ARGS.bdt_prune_ratio
                elif ARGS.defense_type.upper() == "VFLIP":
                    params['threshold'] = ARGS.vflip_threshold
                
                _, _, defense_hooks = build_defense(
                    modelC, optimizerC, defense_type=ARGS.defense_type, **params
                )
                
                if ARGS.defense_type.upper() == "ISO" and defense_hooks.instance:
                    print(f"Adding ISO layer parameters to optimizer with lr={ARGS.iso_lr}")
                    optimizerC.add_param_group({
                        'params': defense_hooks.instance.parameters(),
                        'lr': ARGS.iso_lr
                    })
        
        print(f"{'='*20} Defense Setup Complete {'='*20}\n")
    
    if not schedulers: # If not DPSGD
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ARGS.epochs) for opt in optimizers]
        schedulerC = optim.lr_scheduler.CosineAnnealingLR(optimizerC, T_max=ARGS.epochs)

    # 11. 开始训练
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50)
    
    # 训练循环
    best_accuracy = 0
    best_inference_acc = 0
    best_asr = 0
    best_epoch = 0
    best_combined_score = 0  # 添加综合评分跟踪
    
    # 存储最佳模型对应的所有指标
    best_metrics = {
        'test_acc': 0,
        'inference_acc': 0,
        'asr': 0,
        'epoch': 0,
        'combined_score': 0  # 添加综合评分
    }
    
    # 标记第一个epoch是否已经处理，用于初始化最佳指标
    first_epoch_processed = False
    
    no_improvement_count = 0
    
    # 在Ebkd前的预训练阶段
    print(f"\n{'='*20} 预训练阶段 (1-{ARGS.Ebkd-1}轮) {'='*20}")
    print(f"此阶段专注于提高标签推断准确率，暂不进行后门攻击")
    
    # 启用早停机制
    print(f"Early Stopping: 启用 (patience={ARGS.patience}, 评估指标=0.5*CleanAcc+0.5*ASR)")
    
    for epoch in range(1, ARGS.epochs + 1):
        print(f"\n开始 Epoch {epoch}/{ARGS.epochs}")
        
        # 记录GPU内存使用
        if torch.cuda.is_available():
            print(get_gpu_memory_usage())
        
        # 确定要传递给 train_epoch 的模型和优化器
        train_model_C = defense_hooks.instance if defense_type == 'DPSGD' else modelC
        train_bottom_models = None if defense_type == 'DPSGD' else bottom_models
        train_optimizer_or_optimizers = optimizer if defense_type == 'DPSGD' else optimizers
        train_optimizer_C = None if defense_type == 'DPSGD' else optimizerC

        # 训练
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            train_model_C, train_bottom_models, train_optimizer_or_optimizers, train_optimizer_C, 
            train_loader, epoch, ARGS, 
            adversary_model.label_inference, defense_hooks
        )
        
        # 确定要传递给 test 的模型
        test_model_C = defense_hooks.instance if defense_type == 'DPSGD' else modelC
        test_bottom_models = None if defense_type == 'DPSGD' else bottom_models

        # 测试 - 使用返回的字典格式结果
        test_results = test(
            test_model_C, test_bottom_models, test_loader, 
            is_backdoor=(epoch >= ARGS.Ebkd), epoch=epoch, args=ARGS, defense_hooks=defense_hooks
        )
        
        # 从结果字典中提取值
        test_loss = test_results['loss']
        test_acc = test_results['clean_acc']
        true_asr = test_results['asr']  # 真实的ASR（在测试集上计算）
        
        # 从标签推断模块获取推断准确率
        test_inference_acc = 0.0
        if hasattr(adversary_model, 'label_inference') and adversary_model.label_inference and adversary_model.label_inference.initialized:
            test_inference_acc = train_inference_acc  # 使用训练时计算的推断准确率
        
        print(f"\nEpoch {epoch} Results:")
        print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, ASR {train_asr:.2f}%, Inference Acc {train_inference_acc:.2f}%")
        print(f"Test: Loss {test_loss:.4f}, Acc {test_acc:.2f}%")
        if epoch >= ARGS.Ebkd:
            print(f"Backdoor: ASR {true_asr:.2f}%")
        else:
            print(f"Backdoor attack not active yet (starts at epoch {ARGS.Ebkd})")

        # Check if this is the best model
        # Only consider ASR in combined score if backdoor attack has started
        if epoch >= ARGS.Ebkd:
            combined_score = 0.5 * test_acc + 0.5 * true_asr
            best_combined_score = 0.5 * best_metrics['test_acc'] + 0.5 * best_metrics['asr']
        else:
            # Before backdoor starts, only consider clean accuracy
            combined_score = test_acc
            best_combined_score = best_metrics['test_acc']
        
        is_best = False
        if not first_epoch_processed or combined_score > best_combined_score:
            # 更新最佳指标
            best_metrics['test_acc'] = test_acc
            best_metrics['asr'] = true_asr
            best_metrics['inference_acc'] = test_inference_acc
            best_metrics['epoch'] = epoch
            best_metrics['combined_score'] = combined_score
            
            # 重置early stopping计数器
            no_improvement_count = 0
            is_best = True
            first_epoch_processed = True
            
            print(f"[BEST] New best model! Combined Score: {combined_score:.2f}%")
            
            # 保存最佳模型
            save_model_C = defense_hooks.instance.top_model._module if defense_type == 'DPSGD' else modelC
            save_bottom_models = [m._module for m in defense_hooks.instance.bottom_models] if defense_type == 'DPSGD' else bottom_models
            save_checkpoint(save_model_C, save_bottom_models, epoch, test_acc, true_asr, test_inference_acc)
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs. Best combined score: {best_combined_score:.2f}%")
        
        # Early stopping check
        if ARGS.early_stopping and no_improvement_count >= ARGS.patience:
            # 确保至少训练了Ebkd+5轮，给后门攻击一些时间
            min_epochs_after_backdoor = ARGS.Ebkd + 5 if epoch >= ARGS.Ebkd else ARGS.min_epochs
            if epoch >= min_epochs_after_backdoor:
                print(f"\nEarly stopping triggered! No improvement for {ARGS.patience} epochs.")
                print(f"Best model was at epoch {best_metrics['epoch']}")
                break
            else:
                print(f"Early stopping postponed. Need at least {min_epochs_after_backdoor} epochs (current: {epoch})")
                no_improvement_count = 0  # 重置计数器，继续训练
        
        # Update learning rate
        if defense_type == 'DPSGD':
            schedulers[0].step()
        else:
            for scheduler in schedulers:
                scheduler.step()
            schedulerC.step()

        # 无论如何，一定要让模型训练到Ebkd+10，确保后门攻击有足够轮次
        if epoch == ARGS.Ebkd + 10:
            print(f"\n已完成后门攻击的前10轮训练，目前ASR: {true_asr:.2f}%, 综合评分: {combined_score:.2f}%")
        
        # 每个epoch结束时清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU内存已清理")
            print(get_gpu_memory_usage())
    
    # 训练结束，输出详细的最佳结果
    print("\n" + "="*60)
    print(f"训练完成！最佳模型 (Epoch {best_metrics['epoch']}):")
    print(f"Clean Accuracy: {best_metrics['test_acc']:.2f}%")
    print(f"Attack Success Rate: {best_metrics['asr']:.2f}%")
    print(f"Inference Accuracy: {best_metrics['inference_acc']:.2f}%")
    print(f"Combined Score (0.5*CleanAcc + 0.5*ASR): {best_metrics['combined_score']:.2f}%")
    print("="*60)

# 添加一个测试函数来确认GPU工作正常
def test_gpu_working():
    """测试GPU是否正常工作"""
    if not torch.cuda.is_available():
        print("没有可用的GPU!")
        return False
    
    try:
        # 设置当前设备
        torch.cuda.set_device(ARGS.gpu)
        current_device = torch.cuda.current_device()
        print(f"当前CUDA设备: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # 在GPU上创建一个大张量并执行操作
        x = torch.randn(1000, 1000, device=f"cuda:{current_device}")
        y = torch.randn(1000, 1000, device=f"cuda:{current_device}")
        
        # 强制执行GPU计算并同步
        start_time = time.time()
        z = torch.matmul(x, y)
        # 强制同步以确保计算完成
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 检查计算速度 - GPU应该很快
        elapsed = end_time - start_time
        print(f"GPU矩阵乘法耗时: {elapsed:.6f}秒")
        
        # 在CPU上进行相同计算
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        end_time = time.time()
        
        elapsed_cpu = end_time - start_time
        print(f"CPU矩阵乘法耗时: {elapsed_cpu:.6f}秒")
        
        speedup = elapsed_cpu / elapsed if elapsed > 0 else 0
        print(f"GPU加速比: {speedup:.2f}x")
        
        # 如果GPU比CPU快5倍以上，可能工作正常
        if speedup >= 5:
            print("GPU工作正常!")
            return True
        else:
            print("警告: GPU可能未正常工作，加速比低于预期")
            return False
    
    except Exception as e:
        print(f"GPU测试失败: {str(e)}")
        return False

# 主函数结尾部分
if __name__ == '__main__':
    # 在运行主函数前，先运行一个GPU测试
    print("\n========== GPU测试 ==========")
    import torch
    if torch.cuda.is_available():
        # 因为设置了CUDA_VISIBLE_DEVICES，测试变得更简单
        print(f"CUDA可用。设备数量: {torch.cuda.device_count()} (被CUDA_VISIBLE_DEVICES限制)")
        print(f"使用GPU: {torch.cuda.get_device_name(0)} (物理ID: {ARGS.gpu})")
        
        # 运行一个简单的矩阵乘法测试
        print("运行GPU速度测试...")
        try:
            # 创建大矩阵，直接使用 "cuda"
            a = torch.randn(2000, 2000, device="cuda")
            b = torch.randn(2000, 2000, device="cuda")
            
            # 测量时间
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # 预热
            for _ in range(5):
                c = torch.matmul(a, b)
            
            # 计时
            start.record()
            for _ in range(10):
                c = torch.matmul(a, b)
            end.record()
            
            # 同步GPU
            torch.cuda.synchronize()
            
            # 计算时间
            gpu_time = start.elapsed_time(end) / 10  # 平均每次乘法的时间
            print(f"GPU 矩阵乘法 (2000x2000) 平均时间: {gpu_time:.3f} ms")
            
            # 测试CPU时间进行对比
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            
            import time
            cpu_start = time.time()
            for _ in range(3):  # CPU慢，只做3次
                c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_end = time.time()
            
            cpu_time = (cpu_end - cpu_start) * 1000 / 3  # 毫秒
            print(f"CPU 矩阵乘法 (2000x2000) 平均时间: {cpu_time:.3f} ms")
            
            # 计算加速比
            speedup = cpu_time / gpu_time
            print(f"GPU比CPU快 {speedup:.1f} 倍")
            
            if speedup < 5:
                print("警告: GPU加速不明显，可能存在问题!")
            else:
                print("GPU工作正常!")
                
        except Exception as e:
            print(f"GPU测试失败: {str(e)}")
    else:
        print("CUDA不可用，将使用CPU (运行会非常慢)")
    
    print("==============================\n")
    
    # 运行主函数
    main() 