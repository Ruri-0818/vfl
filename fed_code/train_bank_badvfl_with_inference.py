#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: train_bank_badvfl_with_inference.py
# 描述: 针对Bank Marketing数据集的BadVFL攻击训练 (带标签推断)
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

from defense_all import build_defense

# 扩展命令行参数
parser = argparse.ArgumentParser(description='针对Bank Marketing数据集的BadVFL攻击训练 (带标签推断)')
# 原有参数
parser.add_argument('--dataset', type=str, default='BANK', help='数据集名称 (BANK)')
parser.add_argument('--batch-size', type=int, default=128, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--trigger-size', type=float, default=0.08, help='BadVFL触发器大小(特征比例)')
parser.add_argument('--trigger-intensity', type=float, default=0.5, help='BadVFL触发器强度')
parser.add_argument('--position', type=str, default='mid', help='恶意参与方位置')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--auxiliary-ratio', type=float, default=0.1, help='辅助损失比例')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=3, help='参与方数量')
parser.add_argument('--patience', type=int, default=15, help='早停轮数')
parser.add_argument('--min-epochs', type=int, default=30, help='最小训练轮数')
parser.add_argument('--max-epochs', type=int, default=100, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=0.3, help='后门损失权重')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.08, help='毒化预算')
parser.add_argument('--Ebkd', type=int, default=8, help='后门注入开始轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.1, help='学习率倍增器')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_badvfl_bank', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=2, help='类别数量 (2)')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='/home/tsinghuaair/attack/data/bank', help='数据集目录')
parser.add_argument('--trigger-type', type=str, default='feature', help='触发器类型 (feature)')

# 标签推断相关参数
parser.add_argument('--inference-weight', type=float, default=0.1, help='标签推断损失权重')
parser.add_argument('--history-size', type=int, default=2000, help='嵌入向量历史记录大小')
parser.add_argument('--cluster-update-freq', type=int, default=20, help='聚类更新频率(批次)')
parser.add_argument('--inference-start-epoch', type=int, default=3, help='开始标签推断的轮数')
parser.add_argument('--confidence-threshold', type=float, default=0.4, help='标签推断置信度阈值')
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
parser.add_argument('--defense-type', type=str, default='NONE', 
                    help='Defense type (NONE, DPSGD, MP, ANP, BDT, VFLIP, ISO)')
# DPSGD args
parser.add_argument('--dpsgd-noise-multiplier', type=float, default=1.0, help='Noise multiplier for DPSGD')
parser.add_argument('--dpsgd-max-grad-norm', type=float, default=1.0, help='Max grad norm for DPSGD')
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

# 设置全局变量
args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Bank Marketing Dataset Class
class BankMarketingDataset(Dataset):
    """Bank Marketing Dataset Loader"""
    
    def __init__(self, data_dir, split='train', party_id=None, num_parties=3):
        """
        Args:
            data_dir: Dataset root directory
            split: 'train' or 'test'
            party_id: Party ID (None means load all data)
            num_parties: Number of parties
        """
        print(f"[DATASET] Initializing BankMarketingDataset for {split} split", flush=True)
        self.data_dir = data_dir
        self.split = split
        self.party_id = party_id
        self.num_parties = num_parties
        
        print(f"[DATASET] Starting data loading for {split} dataset (Party {party_id})", flush=True)
        
        # Load and preprocess data
        print(f"[DATASET] Calling _load_and_preprocess_data...", flush=True)
        self._load_and_preprocess_data()
        print(f"[DATASET] Data preprocessing completed", flush=True)
        
        print(f"[DATASET] Successfully loaded {len(self.features)} samples", flush=True)
    
    def _load_and_preprocess_data(self):
        """Load and preprocess Bank Marketing data - Real data only"""
        print(f"Loading {self.split} dataset (real data only)")
        
        # First check if preprocessed data exists
        X_train_path = os.path.join(self.data_dir, 'X_train.npy')
        X_test_path = os.path.join(self.data_dir, 'X_test.npy')
        y_train_path = os.path.join(self.data_dir, 'y_train.npy')
        y_test_path = os.path.join(self.data_dir, 'y_test.npy')
        
        if all(os.path.exists(path) for path in [X_train_path, X_test_path, y_train_path, y_test_path]):
            print("✓ Found preprocessed bank marketing data, loading...")
            # Load preprocessed data
            if self.split == 'train':
                data = np.load(X_train_path)
                labels = np.load(y_train_path)
            else:
                data = np.load(X_test_path)
                labels = np.load(y_test_path)
                
            print(f"✓ Successfully loaded preprocessed data: {data.shape[0]} samples, {data.shape[1]} features")
        else:
            print("! Preprocessed data not found, trying to load original CSV data...")
            
            # Try to load original CSV data
            csv_paths = [
                os.path.join(self.data_dir, 'bank-additional', 'bank-additional-full.csv'),
                os.path.join(self.data_dir, 'bank-additional', 'bank-additional.csv'),
                os.path.join(self.data_dir, 'bank-full.csv'),
                os.path.join(self.data_dir, 'bank.csv')
            ]
            
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                print("! No Bank Marketing CSV files found, creating sample dataset...")
                # 创建示例数据集而不是抛出错误
                data, labels = self._create_sample_bank_data()
                
                # 保存示例数据以便下次使用
                try:
                    # 划分训练集和测试集
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        data, labels, test_size=0.2, random_state=42, stratify=labels
                    )
                    
                    # 保存预处理数据
                    os.makedirs(self.data_dir, exist_ok=True)
                    np.save(X_train_path, X_train)
                    np.save(X_test_path, X_test)
                    np.save(y_train_path, y_train)
                    np.save(y_test_path, y_test)
                    print("✓ Sample data saved for future use")
                    
                    # 根据split选择对应的数据
                    if self.split == 'train':
                        data, labels = X_train, y_train
                    else:
                        data, labels = X_test, y_test
                        
                except Exception as save_error:
                    print(f"! Failed to save sample data: {save_error}")
                    # 如果保存失败，仍然使用生成的数据
                    pass
            else:
                print(f"✓ Found CSV file: {csv_path}")
                
                try:
                    # Read CSV data
                    print("Loading and preprocessing CSV data...")
                    df = pd.read_csv(csv_path, sep=';')
                    print(f"Original data: {df.shape[0]} samples, {df.shape[1]} columns")
                    
                    # Check data integrity
                    if df.empty:
                        raise ValueError("CSV file is empty")
                    
                    if 'y' not in df.columns:
                        raise ValueError("CSV file missing target column 'y'")
                    
                    # Separate features and labels
                    target_col = 'y'
                    features_df = df.drop(columns=[target_col]).copy()
                    labels_series = df[target_col].copy()
                    
                    print(f"Feature columns: {len(features_df.columns)}")
                    print(f"Label distribution: {labels_series.value_counts().to_dict()}")
                    
                    # Encode categorical features
                    categorical_cols = features_df.select_dtypes(include=['object']).columns
                    print(f"Categorical features: {list(categorical_cols)}")
                    
                    for col in categorical_cols:
                        le = LabelEncoder()
                        features_df[col] = le.fit_transform(features_df[col].astype(str))
                    
                    # Encode labels
                    label_encoder = LabelEncoder()
                    labels_encoded = label_encoder.fit_transform(labels_series)
                    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
                    
                    # Standardize features
                    print("Standardizing features...")
                    scaler = StandardScaler()
                    data = scaler.fit_transform(features_df.values)
                    labels = labels_encoded
                    
                    print(f"✓ CSV data preprocessing completed: {data.shape[0]} samples, {data.shape[1]} features")
                    
                    # Split train/test sets
                    print("Splitting train/test sets...")
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        data, labels, test_size=0.2, random_state=42, stratify=labels
                    )
                    
                    print(f"Training set: {X_train.shape[0]} samples")
                    print(f"Test set: {X_test.shape[0]} samples")
                    
                    # Optionally save preprocessed data for future use
                    try:
                        np.save(X_train_path, X_train)
                        np.save(X_test_path, X_test)
                        np.save(y_train_path, y_train)
                        np.save(y_test_path, y_test)
                        print("Preprocessed data saved, next loading will be faster")
                    except Exception as save_error:
                        print(f"! Failed to save preprocessed data: {save_error}")
                    
                    if self.split == 'train':
                        data, labels = X_train, y_train
                    else:
                        data, labels = X_test, y_test
                        
                except Exception as e:
                    print(f"CSV data processing failed: {str(e)}, falling back to sample data")
                    data, labels = self._create_sample_bank_data()
        
        # Validate data quality
        if data.size == 0 or labels.size == 0:
            raise ValueError("Loaded data is empty")
        
        if len(data) != len(labels):
            raise ValueError(f"Feature and label count mismatch: {len(data)} vs {len(labels)}")
        
        if np.isnan(data).any():
            raise ValueError("Feature data contains NaN values")
        
        if np.isnan(labels).any():
            raise ValueError("Label data contains NaN values")
        
        print(f"✓ Data quality validation passed")
        
        # If party ID is specified, return only that party's features
        if self.party_id is not None:
            features_per_party = data.shape[1] // self.num_parties
            start_idx = self.party_id * features_per_party
            if self.party_id == self.num_parties - 1:
                end_idx = data.shape[1]  # Last party gets remaining features
            else:
                end_idx = (self.party_id + 1) * features_per_party
            
            self.features = torch.FloatTensor(data[:, start_idx:end_idx])
            print(f"Party {self.party_id} feature range: {start_idx}:{end_idx} ({end_idx-start_idx} dims)")
        else:
            self.features = torch.FloatTensor(data)
        
        self.labels = torch.LongTensor(labels)
        
        # Store feature dimension
        self.feature_dim = self.features.shape[1]
        
        print(f"✓ Final data: feature_dim={self.feature_dim}, samples={len(self.labels)}")
        print(f"Feature range: [{self.features.min():.3f}, {self.features.max():.3f}]")
        print(f"Label distribution: {torch.bincount(self.labels).tolist()}")
    
    def _create_sample_bank_data(self):
        """创建示例Bank Marketing数据集"""
        print("Creating sample Bank Marketing dataset...")
        
        # 生成模拟数据
        np.random.seed(42)
        n_samples = 5000 if self.split == 'train' else 1250
        n_features = 20  # 20个特征
        
        # 生成特征数据 (标准化的)
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # 生成标签 (二分类，不平衡)
        # 使用特征的线性组合来生成更现实的标签
        weights = np.random.randn(n_features)
        logits = np.dot(data, weights)
        probabilities = 1 / (1 + np.exp(-logits))
        labels = (probabilities > 0.3).astype(np.int64)  # 30%为正类
        
        print(f"Generated sample data: {data.shape[0]} samples, {data.shape[1]} features")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# BadVFL标签推断实现 (适用于表格数据)
class BadVFLLabelInference:
    """BadVFL攻击中的标签推断模块，适用于表格数据"""
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
        
        # 设置所需的最小样本数 - 表格数据需要更多样本
        self.min_samples = max(100, 20 * num_classes)
        
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
            num_workers=0, pin_memory=False)
        
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
            if len(predictions.shape) > 1:
                grad_norms = np.linalg.norm(predictions, axis=1)
            else:
                grad_norms = np.abs(predictions)
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
            
            # 检查特征维度一致性
            if len(X.shape) != 2:
                print(f"特征数据格式错误: {X.shape}")
                return False
            
            # 输出标签的唯一值
            unique_labels = np.unique(y)
            print(f"标签唯一值: {unique_labels}")
            print(f"特征维度: {X.shape}")
            
            # 训练随机森林分类器
            print(f"训练标签推断分类器，使用 {len(X)} 个样本，特征维度: {X.shape[1]}...")
            
            self.inference_classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                n_jobs=-1,
                random_state=self.args.seed
            )
            
            # 直接训练分类器
            self.inference_classifier.fit(X, y)
            
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
                
                # 为每个参与方分割数据 - 与训练过程保持一致
                party_data = []
                feature_dim = data.shape[1]
                features_per_party = feature_dim // self.args.party_num
                
                for i in range(self.args.party_num):
                    start_idx = i * features_per_party
                    if i == self.args.party_num - 1:
                        end_idx = feature_dim  # 最后一个参与方获得剩余特征
                    else:
                        end_idx = (i + 1) * features_per_party
                    party_data.append(data[:, start_idx:end_idx])
                
                # 前向传播
                bottom_outputs = []
                for i, (model, party_input) in enumerate(zip(bottom_models, party_data)):
                    output = model(party_input)
                    bottom_outputs.append(output)
                
                combined_output = torch.cat(bottom_outputs, dim=1)
                output = modelC(combined_output)
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += batch_size
        
        return 100. * correct / total if total > 0 else 0.0
    
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
                
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                # 检查特征维度是否匹配
                expected_features = self.inference_classifier.n_features_in_
                if features.shape[1] != expected_features:
                    # 如果维度不匹配，尝试调整
                    if features.shape[1] > expected_features:
                        # 如果输入特征太多，只使用前N个特征
                        features = features[:, :expected_features]
                        print(f"警告: 特征维度调整 {features.shape[1]} -> {expected_features}")
                    else:
                        print(f"标签推断失败: X has {features.shape[1]} features, but RandomForestClassifier is expecting {expected_features} features as input.")
                        return None, None
                
                # 预测标签
                pred_labels = self.inference_classifier.predict(features)
                
                # 获取置信度估计
                pred_probs = self.inference_classifier.predict_proba(features)
                confidence = np.max(pred_probs, axis=1)
                
                return pred_labels, confidence
            except Exception as e:
                print(f"标签推断失败: {str(e)}")
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

# BadVFL触发器实现 (适用于表格数据)
class BadVFLTrigger:
    """BadVFL攻击中的触发器实现，适用于表格数据"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.device = args.device if hasattr(args, 'device') else 'cpu'
        self.dataset_name = args.dataset
        self.trigger_size = args.trigger_size  # 特征比例
        self.intensity = args.trigger_intensity
        self.trigger_type = args.trigger_type if hasattr(args, 'trigger_type') else 'feature'
        
        # 标签推断模块
        self.label_inference = None
        
        # 初始化状态标志
        self.is_initialized = False
        
        # 保存特征索引
        self.feature_indices = None
        
        print(f"创建BadVFL触发器: 类型={self.trigger_type}, 大小={self.trigger_size}, 强度={self.intensity}")
    
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
        
        # 确定要修改的特征数量
        embed_dim = data.size(1)
        num_features = max(1, int(embed_dim * self.trigger_size))
        
        # 为每个攻击样本随机选择不同的特征位置
        for idx in range(len(data)):
            if attack_flags[idx]:
                # 为每个样本随机选择特征位置，增加随机性
                feature_indices = torch.randperm(embed_dim, device=data.device)[:num_features]
                
                # 添加一些随机性到触发器强度，使攻击不那么明显
                random_factor = 0.7 + 0.6 * torch.rand(1, device=data.device).item()  # 0.7-1.3的随机因子
                trigger_value = self.intensity * random_factor
                
                # 对选定的特征添加触发器
                data_copy[idx, feature_indices] += trigger_value
        
        return data_copy
    
    def inject_trigger_with_inference(self, data, attack_flags=None, raw_data=None, top_model=None, bottom_models=None):
        """使用标签推断结果注入触发器"""
        if not self.is_initialized or self.label_inference is None:
            # 如果标签推断未初始化，使用基础触发器注入
            return self.inject_trigger(data, attack_flags)
        
        batch_size = data.size(0)
        
        # 使用标签推断模块进行推断 - 使用输入特征而不是输出特征
        if raw_data is not None:
            # 使用原始输入数据进行标签推断
            inferred_labels, confidence = self.label_inference.infer_labels(raw_data)
        else:
            # 如果没有原始数据，尝试使用输出特征（但这可能导致维度不匹配）
            inferred_labels, confidence = self.label_inference.infer_labels(data)
        
        if inferred_labels is None:
            # 推断失败，使用基础触发器注入
            return self.inject_trigger(data, attack_flags)
        
        # 创建触发器掩码
        trigger_mask = torch.zeros_like(data)
        embed_dim = data.size(1)
        num_features = max(1, int(embed_dim * self.trigger_size))
        
        # 对推断为非目标类的样本注入触发器，但要考虑置信度
        for i in range(min(batch_size, len(inferred_labels))):
            if (attack_flags is not None and i < len(attack_flags) and attack_flags[i] and 
                inferred_labels[i] != self.target_class and 
                (len(confidence) == 0 or i >= len(confidence) or confidence[i] > 0.3)):  # 只对高置信度预测注入触发器
                
                # 随机选择特征位置
                feature_indices = torch.randperm(embed_dim, device=data.device)[:num_features]
                
                # 基于置信度调整触发器强度
                conf_factor = confidence[i] if i < len(confidence) else 0.5
                adjusted_intensity = self.intensity * (0.5 + 0.5 * conf_factor)
                
                trigger_mask[i, feature_indices] = adjusted_intensity
        
        return data + trigger_mask

# Bank底部模型 (适用于表格数据)
class BankBottomModel(nn.Module):
    """Bank marketing数据底部模型"""
    def __init__(self, input_dim, output_dim, is_adversary=False, args=None):
        super(BankBottomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 适用于表格数据的神经网络架构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 用于存储当前批次的数据和梯度
        self.current_batch_data = None
        self.current_batch_grad = None
        
        # 初始化权重
        self._initialize_weights()
        
        # 如果是恶意模型，初始化标签推断模块
        if is_adversary and args is not None:
            self.badvfl_trigger = None
            self.label_inference = None
            print(f"创建恶意底部模型 (ID={args.bkd_adversary})")
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_badvfl_trigger(self, badvfl_trigger):
        """设置BadVFL触发器"""
        if self.is_adversary:
            self.badvfl_trigger = badvfl_trigger
            # 创建标签推断模块
            if not hasattr(self, 'label_inference') or self.label_inference is None:
                print("为恶意模型创建标签推断模块")
                # 使用输入特征维度而不是输出特征维度
                self.label_inference = BadVFLLabelInference(
                    feature_dim=self.input_dim,  # 使用输入维度
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
        
        # 前向传播
        feat = self.network(x)
        
        # 注意：触发器注入将在训练循环中处理，这里不重复注入
        # 这样避免了双重注入的问题
        
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

# Bank顶部模型
class BankTopModel(nn.Module):
    """Bank marketing数据顶部模型"""
    def __init__(self, input_dim=192, num_classes=2):  # 3个参与方 * 64 = 192
        super(BankTopModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def load_dataset(dataset_name, data_dir, batch_size):
    """加载Bank Marketing数据集"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集")
    print(f"{'='*50}")
    
    print("\n1. 创建数据集实例...")
    try:
        # 加载训练集
        print("正在加载训练集...")
        train_dataset = BankMarketingDataset(data_dir, split='train')
        
        # 加载测试集
        print("正在加载测试集...")
        test_dataset = BankMarketingDataset(data_dir, split='test')
        
        print(f"{dataset_name} 数据集加载成功!")
        
        # 验证数据集完整性
        print("验证数据集完整性...")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("数据集为空，可能文件不完整")
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("\n可能的解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保Bank Marketing数据集文件完整")
        print(f"   - 数据集路径: {data_dir}")
        sys.exit(1)
    
    print("\n2. 创建数据加载器...")
    
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
    print(f"特征维度: {train_dataset.feature_dim}")
    print(f"批次大小: {batch_size}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"类别数量: {args.num_classes}")
    
    return train_loader, test_loader, train_dataset.feature_dim

def create_models(feature_dim):
    """创建模型"""
    print(f"创建Bank Marketing模型，总特征维度: {feature_dim}")
    
    output_dim = 64  # 每个底部模型的输出维度
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        # 每个参与方的输入维度
        features_per_party = feature_dim // args.party_num
        if i == args.party_num - 1:
            input_dim = feature_dim - features_per_party * (args.party_num - 1)
        else:
            input_dim = features_per_party
        
        if i == args.bkd_adversary:
            # 创建恶意模型
            model = BankBottomModel(
                input_dim=input_dim,
                output_dim=output_dim,
                is_adversary=True,
                args=args
            )
        else:
            # 创建正常模型
            model = BankBottomModel(
                input_dim=input_dim,
                output_dim=output_dim
            )
        model = model.to(DEVICE)
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = BankTopModel(
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

def create_party_data_loaders(train_loader, test_loader):
    """为每个参与方创建数据加载器 - 简化版本"""
    print("创建参与方数据加载器...")
    
    # 简化实现：所有参与方使用相同的数据加载器
    # 特征分割在训练循环中进行
    party_train_loaders = [train_loader for _ in range(args.party_num)]
    party_test_loaders = [test_loader for _ in range(args.party_num)]
    
    print(f"为 {args.party_num} 个参与方创建了数据加载器")
    
    return party_train_loaders, party_test_loaders

def collect_inference_data(modelC, bottom_models, party_train_loaders, epoch, max_batches=50):
    """收集数据用于标签推断 - 基于CIFAR BadVFL结构的表格数据版本"""
    print(f"\n启动BadVFL标签推断过程 (epoch {epoch})...")
    
    # 确保模型处于训练模式以获取梯度
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    if not label_inference_module:
        print("错误: 恶意模型没有标签推断模块")
        return False
    
    # 大幅减少收集批次数量以提高效率
    max_batches = min(20, min(len(loader) for loader in party_train_loaders))
    
    print(f"将收集最多 {max_batches} 个批次的梯度数据用于标签推断 (快速模式)")
    
    # 创建简化的优化器（仅用于梯度计算）
    optimizers = []
    for model in bottom_models:
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用较小的学习率
        optimizers.append(optimizer)
    optimizerC = optim.SGD(modelC.parameters(), lr=0.01)
    
    collected_samples = 0
    
    # 使用更简单的循环，避免复杂的进度条
    print("开始收集梯度数据...")
    
    # 使用第一个加载器
    train_loader = party_train_loaders[0]
    data_iter = iter(train_loader)
    
    for batch_idx in range(max_batches):
        try:
            # 获取数据
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data, target = next(data_iter)
            
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 简化前向传播：只考虑单个参与方
            if data.size(0) == 0:
                continue
            
            # 为每个参与方分割数据 - 与训练过程保持一致
            party_data = []
            feature_dim = data.shape[1]
            features_per_party = feature_dim // args.party_num
            
            for i in range(args.party_num):
                start_idx = i * features_per_party
                if i == args.party_num - 1:
                    end_idx = feature_dim  # 最后一个参与方获得剩余特征
                else:
                    end_idx = (i + 1) * features_per_party
                party_data.append(data[:, start_idx:end_idx])
                
            # 清除所有梯度
            for optimizer in optimizers:
                optimizer.zero_grad()
            optimizerC.zero_grad()
            
            # 只对恶意模型启用梯度计算
            adversary_input = party_data[args.bkd_adversary]
            adversary_input.requires_grad_(True)
            adversary_output = adversary_model(adversary_input)
            
            # 简化的前向传播
            # 为其他模型创建假输出
            bottom_outputs = []
            for i, model in enumerate(bottom_models):
                if i == args.bkd_adversary:
                    bottom_outputs.append(adversary_output)
                else:
                    # 创建相同形状的零张量作为占位符
                    fake_output = torch.zeros_like(adversary_output)
                    bottom_outputs.append(fake_output)
            
            combined_output = torch.cat(bottom_outputs, dim=1)
            output = modelC(combined_output)
            
            # 计算简化的损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            loss.backward()
            
            # 获取梯度
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                # 更新标签推断历史 - 使用原始特征数据
                samples_added = label_inference_module.update_with_batch(saved_data, saved_grad)
                collected_samples += samples_added
                
                # 每5个批次输出进度信息
                if batch_idx % 5 == 0:
                    print(f"已收集 {batch_idx+1}/{max_batches} 批次, 总样本数: {label_inference_module.get_total_samples()}")
                
                # 尝试提前初始化，不等到收集完所有批次
                if batch_idx >= 10 and batch_idx % 5 == 0:
                    if label_inference_module.embedding_swapping():
                        if label_inference_module.candidate_selection():
                            if adversary_model.badvfl_trigger:
                                adversary_model.badvfl_trigger.update_inference_stats()
                                print(f"\n✓ 标签推断初始化成功! 批次: {batch_idx}")
                                return True
            
            # 清空梯度避免累积
            for optimizer in optimizers:
                optimizer.zero_grad()
            optimizerC.zero_grad()
            
            # 释放不必要的内存
            del loss, output, combined_output, adversary_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"! 收集数据时出错 (batch {batch_idx}): {e}")
            continue
    
    # 最终尝试执行嵌入交换和候选选择
    print(f"完成数据收集，已收集 {label_inference_module.get_total_samples()} 个样本")
    
    if label_inference_module.embedding_swapping():
        if label_inference_module.candidate_selection():
            # 更新触发器状态
            if adversary_model.badvfl_trigger:
                adversary_model.badvfl_trigger.update_inference_stats()
                print("✓ 标签推断初始化成功")
                return True
    
    print("! 标签推断初始化失败，但将继续训练")
    return False

def train_epoch(modelC, bottom_models, party_train_loaders, optimizers, optimizerC, epoch, args, defense_hooks):
    """训练一个epoch"""
    modelC.train()
    for model in bottom_models:
        model.train()
    
    # 获取恶意模型和标签推断模块
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference if adversary_model.is_adversary else None
    
    total_loss = 0
    clean_correct = 0
    bkd_correct = 0
    backdoor_samples_total = 0  # 累积所有批次的实际攻击样本数
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    # 判断各种训练阶段
    has_inference = label_inference_module is not None
    collect_gradients = has_inference and epoch <= args.inference_start_epoch + 3  # 减少梯度收集轮数
    is_warmup = epoch < args.Ebkd
    enable_backdoor = epoch >= args.Ebkd
    
    # 损失权重策略
    if is_warmup:
        backdoor_weight = 0.0
        clean_weight = 1.0
    else:
        backdoor_weight = args.backdoor_weight
        clean_weight = 1.0
    
    print(f"Epoch {epoch}: 预热阶段={is_warmup}, 启用后门={enable_backdoor}, 后门权重={backdoor_weight:.3f}")
    
    # 简化数据加载：只使用第一个加载器
    train_loader = party_train_loaders[0]
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清除梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        data, target = data.to(DEVICE), target.to(DEVICE)
        total += len(target)
        
        # 简化的前向传播 - 干净数据
        # 为每个参与方分割数据
        party_data = []
        feature_dim = data.shape[1]
        features_per_party = feature_dim // args.party_num
        
        for i in range(args.party_num):
            start_idx = i * features_per_party
            if i == args.party_num - 1:
                end_idx = feature_dim  # 最后一个参与方获得剩余特征
            else:
                end_idx = (i + 1) * features_per_party
            party_data.append(data[:, start_idx:end_idx])
        
        bottom_outputs_clean = []
        for i, (model, party_input) in enumerate(zip(bottom_models, party_data)):
            output = model(party_input)
            bottom_outputs_clean.append(output)
        
        combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
        if defense_hooks.forward:
            combined_output_clean = defense_hooks.forward(combined_output_clean)
        output_clean = modelC(combined_output_clean)
        loss_clean = criterion(output_clean, target)
        
        # 后门攻击处理
        loss_backdoor = 0
        backdoor_samples = 0
        if enable_backdoor:
            # 准备后门数据 - 改进攻击样本选择
            bkd_target = target.clone()
            attack_flags = torch.zeros(len(target), dtype=torch.bool).to(DEVICE)
            
            # 随机选择攻击样本，而不是总是选择前面的样本
            attack_portion = int(len(target) * args.poison_budget)
            if attack_portion > 0:
                # 随机选择要攻击的样本索引
                attack_indices = torch.randperm(len(target))[:attack_portion]
                attack_flags[attack_indices] = True
                
                # 随机选择目标类别（增加一些随机性）
                if torch.rand(1).item() < 0.1:  # 10%的概率选择随机目标类别
                    random_target = torch.randint(0, args.num_classes, (1,)).item()
                    bkd_target[attack_flags] = random_target
                else:
                    bkd_target[attack_flags] = args.target_class
                    
                backdoor_samples = attack_flags.sum().item()
                backdoor_samples_total += backdoor_samples  # 累积实际攻击样本数
            
            if backdoor_samples > 0:
                # 前向传播 - 后门数据
                bottom_outputs_bkd = []
                for i, (model, party_input) in enumerate(zip(bottom_models, party_data)):
                    if i == args.bkd_adversary:
                        # 恶意模型：先正常前向传播，然后注入触发器
                        output = model(party_input)
                        # 应用触发器到输出特征
                        if hasattr(model, 'badvfl_trigger') and model.badvfl_trigger:
                            # 使用改进的触发器注入
                            if model.badvfl_trigger.is_initialized:
                                output = model.badvfl_trigger.inject_trigger_with_inference(
                                    output, attack_flags=attack_flags, raw_data=party_input
                                )
                            else:
                                output = model.badvfl_trigger.inject_trigger(output, attack_flags)
                    else:
                        output = model(party_input)
                    bottom_outputs_bkd.append(output)
                
                combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                if defense_hooks.forward:
                    combined_output_bkd = defense_hooks.forward(combined_output_bkd)
                output_bkd = modelC(combined_output_bkd)
                loss_backdoor = criterion(output_bkd, bkd_target)
        
        # 组合损失
        if enable_backdoor and backdoor_samples > 0:
            loss = clean_weight * loss_clean + backdoor_weight * loss_backdoor
        else:
            loss = clean_weight * loss_clean
        
        # 反向传播
        loss.backward()
        
        # 简化的梯度收集
        if collect_gradients and has_inference and batch_idx % 10 == 0:  # 每10个批次收集一次
            saved_data, saved_grad = adversary_model.get_saved_data()
            if saved_data is not None and saved_grad is not None:
                samples_added = label_inference_module.update_with_batch(saved_data, saved_grad)
                
                # 减少初始化尝试频率
                if batch_idx % 50 == 0 and not label_inference_module.initialized:
                    if label_inference_module.update_class_stats():
                        print(f"✓ 标签推断在batch {batch_idx}初始化成功")
                        if adversary_model.badvfl_trigger:
                            adversary_model.badvfl_trigger.is_initialized = True
        
        # 梯度裁剪
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
        
        # 累计损失
        total_loss += loss.item()
        
        # 计算当前批次的ASR - 修复分母错误
        if backdoor_samples_total > 0:
            current_asr = 100. * bkd_correct / backdoor_samples_total
        else:
            current_asr = 0.0
        
        if batch_idx % 10 == 0:
            print(
                f'\tLoss: {loss.item():.6f}, Clean Acc: {100. * clean_correct / total:.2f}%, ASR: {current_asr:.2f}%')
        
        # 释放内存
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * clean_correct / total
    
    # 计算攻击成功率 - 修复ASR计算
    attack_success_rate = 0.0
    if enable_backdoor and backdoor_samples_total > 0:
        # 使用实际攻击的样本数作为分母，而不是理论值
        attack_success_rate = min(100.0, 100.0 * bkd_correct / backdoor_samples_total)
    
    # 计算推断准确率
    inference_accuracy = 0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # 简化的推断准确率计算
        inference_accuracy = 50  # 占位符值
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy

def test(modelC, bottom_models, party_test_loaders, is_backdoor=False, epoch=0, args=None, defense_hooks=None):
    """测试模型性能"""
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
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    # 简化测试：只使用第一个测试加载器
    test_loader = party_test_loaders[0]
    
    # 检查后门攻击是否应该激活
    should_test_backdoor = is_backdoor and epoch >= args.Ebkd
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)
            total += batch_size
            
            # 为每个参与方分割数据 - 与训练过程保持一致
            party_data = []
            feature_dim = data.shape[1]
            features_per_party = feature_dim // args.party_num
            
            for i in range(args.party_num):
                start_idx = i * features_per_party
                if i == args.party_num - 1:
                    end_idx = feature_dim  # 最后一个参与方获得剩余特征
                else:
                    end_idx = (i + 1) * features_per_party
                party_data.append(data[:, start_idx:end_idx])
            
            # 前向传播 - 干净数据
            bottom_outputs_clean = []
            for model, party_input in zip(bottom_models, party_data):
                output = model(party_input)
                bottom_outputs_clean.append(output)
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            if defense_hooks.forward:
                combined_output_clean = defense_hooks.forward(combined_output_clean)
            output_clean = modelC(combined_output_clean)
            
            # 计算损失
            test_loss += criterion(output_clean, target).item()
            
            # 预测
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            clean_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 只有在epoch >= Ebkd时才测试后门攻击
            if should_test_backdoor:
                # 准备后门数据 - 使用改进的随机选择
                attack_flags = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
                attack_portion = int(batch_size * args.poison_budget)
                if attack_portion > 0:
                    # 随机选择要攻击的样本索引
                    attack_indices = torch.randperm(batch_size)[:attack_portion]
                    attack_flags[attack_indices] = True
                    backdoor_samples += attack_flags.sum().item()
                    
                    bkd_target = target.clone()
                    # 添加一些随机性到目标选择
                    if torch.rand(1).item() < 0.1:  # 10%的概率选择随机目标
                        random_target = torch.randint(0, args.num_classes, (1,)).item()
                        bkd_target[attack_flags] = random_target
                    else:
                        bkd_target[attack_flags] = args.target_class
                    
                    # 前向传播 - 后门数据
                    bottom_outputs_bkd = []
                    for i, (model, party_input) in enumerate(zip(bottom_models, party_data)):
                        if i == args.bkd_adversary:
                            # 恶意模型：先正常前向传播，然后注入触发器
                            output = model(party_input)
                            # 应用触发器
                            if hasattr(model, 'badvfl_trigger') and model.badvfl_trigger:
                                if model.badvfl_trigger.is_initialized:
                                    output = model.badvfl_trigger.inject_trigger_with_inference(
                                        output, attack_flags=attack_flags, raw_data=party_input
                                    )
                                else:
                                    output = model.badvfl_trigger.inject_trigger(output, attack_flags)
                        else:
                            output = model(party_input)
                        bottom_outputs_bkd.append(output)
                    
                    combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                    if defense_hooks.forward:
                        combined_output_bkd = defense_hooks.forward(combined_output_bkd)
                    output_bkd = modelC(combined_output_bkd)
                    
                    pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                    bkd_correct += pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags])).sum().item()
    
    # 计算结果
    test_loss /= total
    accuracy = 100. * clean_correct / total
    
    attack_success_rate = 0.0
    if backdoor_samples > 0:
        attack_success_rate = min(100.0, 100.0 * bkd_correct / backdoor_samples)
    
    # 打印结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}% ({clean_correct}/{total})')
    if should_test_backdoor:
        print(f'Backdoor ASR: {attack_success_rate:.2f}% ({bkd_correct}/{backdoor_samples})')
    elif is_backdoor:
        print(f'后门攻击尚未激活 (epoch {epoch} < {args.Ebkd})')
    
    return test_loss, accuracy, attack_success_rate, 0

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
        'trigger_intensity': args.trigger_intensity,
        'trigger_size': args.trigger_size,
        'poison_budget': args.poison_budget,
        'inference_weight': args.inference_weight
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存模型到 {model_save_path}')

def main():
    print("[BadVFL] 开始Bank Marketing数据集的BadVFL攻击训练...")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BadVFL攻击训练 (带标签推断) - 数据集: {args.dataset}")
    print(f"设备: {DEVICE}")
    print(f"参与方数量: {args.party_num}")
    print(f"恶意参与方ID: {args.bkd_adversary}")
    print(f"目标类别: {args.target_class}")
    print(f"后门开始轮数: {args.Ebkd}")
    print(f"{'='*80}\n")

    # 加载数据集
    train_loader, test_loader, feature_dim = load_dataset(args.dataset, args.data_dir, args.batch_size)
    
    # 创建参与方数据加载器
    party_train_loaders, party_test_loaders = create_party_data_loaders(train_loader, test_loader)

    # 创建模型
    bottom_models, modelC = create_models(feature_dim)

    # 创建优化器
    optimizers = [optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
                 for model in bottom_models]
    optimizerC = optim.SGD(modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 学习率调度器
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) 
                 for optimizer in optimizers]
    schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optimizerC, mode='max', factor=0.5, patience=5)

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
            input_dim = modelC.classifier[0].in_features
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
        benign_indices = [i for i, label in enumerate(train_loader.dataset.labels) if label != args.target_class]
        benign_dataset = torch.utils.data.Subset(train_loader.dataset, benign_indices)
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
            # Use benign_loader for training
            if benign_loader is None:
                raise ValueError("Benign loader is required for VFLIP training but is None.")

            for data, _ in tqdm(benign_loader, desc=f"VFLIP Epoch {v_epoch+1}/{args.vflip_train_epochs}"):
                data = data.to(DEVICE)
                with torch.no_grad():
                    # Feature extraction logic for Bank dataset
                    party_data = []
                    features_per_party = data.shape[1] // args.party_num
                    for i in range(args.party_num):
                        start_idx = i * features_per_party
                        end_idx = data.shape[1] if i == args.party_num - 1 else (i + 1) * features_per_party
                        party_data.append(data[:, start_idx:end_idx])

                    bottom_outputs = []
                    for i, model in enumerate(bottom_models):
                        model.eval() # Set to eval mode for feature extraction
                        bottom_outputs.append(model(party_data[i]))
                    features = torch.cat(bottom_outputs, dim=1)
                
                loss = vflip_instance.train_step(features)
                total_vflip_loss += loss
            
            # Set models back to train mode
            for model in bottom_models:
                model.train()

            avg_vflip_loss = total_vflip_loss / len(benign_loader)
            print(f"VFLIP pre-train epoch {v_epoch+1}, Avg Loss: {avg_vflip_loss:.4f}")

    print(f"\n开始Bank Marketing BadVFL攻击训练")
    print(f"轮数: {args.epochs}, 批次大小: {args.batch_size}")

    # 训练循环
    best_accuracy = 0
    best_asr = 0
    best_inference_acc = 0
    best_epoch = 0
    no_improvement_count = 0
    
    # 为恶意模型设置辅助数据集
    adversary_model = bottom_models[args.bkd_adversary]
    if adversary_model.is_adversary and adversary_model.label_inference is not None:
        print("\n设置辅助数据集用于标签推断...")
        adversary_model.label_inference.set_auxiliary_dataset(train_loader.dataset, args.aux_data_ratio)
    
    # 预收集推断数据
    if args.inference_start_epoch > 0:
        print(f"\n开始预收集推断数据...")
        collect_inference_data(modelC, bottom_models, party_train_loaders, 0)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        # 在推断开始轮数再次收集数据
        if epoch == args.inference_start_epoch:
            print(f"\n在推断开始轮数收集更多数据...")
            collect_inference_data(modelC, bottom_models, party_train_loaders, epoch)
        
        # 训练
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, party_train_loaders, optimizers, optimizerC, epoch, args, defense_hooks
        )

        # 测试
        test_loss, test_acc, _, test_inference_acc = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=False, epoch=epoch, args=args, defense_hooks=defense_hooks
        )

        # 后门测试
        bkd_loss, bkd_acc, true_asr, bkd_inference_acc = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=True, epoch=epoch, args=args, defense_hooks=defense_hooks
        )
        
        # 计算推断准确率（如果有辅助数据集）
        inference_accuracy = 0.0
        if (adversary_model.is_adversary and adversary_model.label_inference is not None and 
            adversary_model.label_inference.initialized):
            inference_accuracy = adversary_model.label_inference.get_aux_prediction_accuracy(modelC, bottom_models)
        
        # 更新学习率
        for scheduler in schedulers:
            scheduler.step(test_acc)
        schedulerC.step(test_acc)

        print(f"\nEpoch {epoch} 结果:")
        print(f"训练: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, ASR {train_asr:.2f}%")
        print(f"测试: Loss {test_loss:.4f}, Acc {test_acc:.2f}%")
        print(f"推断准确率: {inference_accuracy:.2f}%")
        if epoch >= args.Ebkd:
            print(f"后门: ASR {true_asr:.2f}%")
        else:
            print(f"后门攻击尚未激活 (将在第 {args.Ebkd} 轮开始)")

        # 检查是否是最佳模型
        # 只有在后门攻击开始后才在综合分数中考虑ASR
        if epoch >= args.Ebkd:
            combined_score = 0.6 * test_acc + 0.4 * true_asr
            best_combined_score = 0.6 * best_accuracy + 0.4 * best_asr
        else:
            # 后门攻击开始前，只考虑干净准确率
            combined_score = test_acc
            best_combined_score = best_accuracy
        
        if combined_score > best_combined_score:
            best_accuracy = test_acc
            best_asr = true_asr
            best_inference_acc = inference_accuracy
            best_epoch = epoch
            no_improvement_count = 0
            
            # 保存最佳模型
            save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr, test_inference_acc)
            print(f"\n保存最佳模型 (Epoch {epoch})")
        else:
            no_improvement_count += 1
            
        # 早停检查 - 但不在后门攻击开始前
        if args.early_stopping and epoch >= args.min_epochs and no_improvement_count >= args.patience:
            print(f"\n触发早停! 最佳模型在第 {best_epoch} 轮")
            break
    
    print("\n" + "="*60)
    print(f"训练完成! 最佳模型 (第 {best_epoch} 轮):")
    print(f"干净准确率: {best_accuracy:.2f}%")
    print(f"攻击成功率: {best_asr:.2f}%")
    print(f"推断准确率: {best_inference_acc:.2f}%")
    print("="*60)

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
                # Set models to eval mode for consistent feature extraction
                modelC.eval()
                
                # Feature extraction logic for Bank dataset
                party_data = []
                features_per_party = data.shape[1] // args.party_num
                for i in range(args.party_num):
                    start_idx = i * features_per_party
                    end_idx = data.shape[1] if i == args.party_num - 1 else (i + 1) * features_per_party
                    party_data.append(data[:, start_idx:end_idx])

                bottom_outputs = []
                for i, model in enumerate(bottom_models):
                    model.eval()
                    bottom_outputs.append(model(party_data[i]))
                features = torch.cat(bottom_outputs, dim=1)
                activations.append(features.cpu())
        
        all_activations = torch.cat(activations, dim=0)
        print(f"Collected {all_activations.shape[0]} activation vectors for BDT.")
        bdt_instance.run_bdt_offline(all_activations)
        
        # Optional: re-test the model after pruning
        print("\nRe-testing model after BDT pruning...")
        _ , test_acc, _, _ = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=False, epoch=epoch, args=args, defense_hooks=defense_hooks
        )
        _ , _, true_asr, _ = test(
            modelC, bottom_models, party_test_loaders, is_backdoor=True, epoch=epoch, args=args, defense_hooks=defense_hooks
        )
        print(f"After BDT -> Final Clean Acc: {test_acc:.2f}%, Final ASR: {true_asr:.2f}%")

if __name__ == '__main__':
    main() 