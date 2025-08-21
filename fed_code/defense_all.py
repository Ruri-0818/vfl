#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Defense Library (Standalone)
=================================================
本文件自包含六种防御实现，可直接 `import defense_all` 并调用
    • DPSGD – 微分隐私随机梯度下降 (Opacus)
    • MP    – 全局 L1 权值剪枝
    • ANP   – 激活高斯噪声扰动
    • BDT   – 光谱签名检测 + 精剪 (简化)
    • VFLIP – 受论文启发的特征空间异常检测 (内置自编码器)
    • ISO   – 特征空间隔离层 (新增)

调用示例:
    from defense_all import build_defense
    # 1. 构建防御
    #    - 对于 VFLIP/BDT/ISO, 返回的 hooks 对象会包含防御实例 `hooks.instance`
    #    - 用户需要用此实例来执行额外的训练或分析步骤
    model, optimizer, hooks = build_defense(
        model, optimizer,
        defense_type='VFLIP',          # 或 NONE/DPSGD/MP/ANP/BDT/ISO
        # --- 各防御参数 ---
        batch_size=128, sample_size=len(train_dataset), noise_multiplier=1.0, # DPSGD
        input_dim=1024,                # VFLIP / ISO
        pruning_amount=0.2,            # MP
        sigma=0.1,                     # ANP
        prune_ratio=0.2)               # BDT

    # 2. 在训练循环中使用
    # A. VFLIP: 需要在良性数据上预训练MAE
    # if hooks.instance and defense_type == 'VFLIP':
    #     for features in benign_dataloader:
    #         hooks.instance.train_step(features)

    # B. BDT: 需要在训练后，用良性数据进行分析和剪枝
    # if hooks.instance and defense_type == 'BDT':
    #     activations = get_activations(benign_dataloader)
    #     hooks.instance.run_bdt_offline(activations)

    # C. ISO: 需要和主模型一起训练
    # if hooks.instance and defense_type == 'ISO':
    #     iso_optimizer.zero_grad()
    #     iso_loss.backward()
    #     iso_optimizer.step()

    # D. 通用前向传播钩子 (VFLIP, ANP, ISO)
    feats = torch.cat(bottom_outputs, dim=1)
    if hooks.forward:
        feats = hooks.forward(feats)
    pred = top_model(feats)
"""
from __future__ import annotations
import logging
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)

###############################################################################
# DPSGD -----------------------------------------------------------------------
###############################################################################
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
    # Add a version check for RMSNorm for safety, as new opacus depends on new torch
    if not hasattr(torch.nn, 'RMSNorm'):
        print("Warning: Your PyTorch version might be older than 2.1. DPSGD defense may be incompatible. Disabling DPSGD.")
        OPACUS_AVAILABLE = False
except ImportError:
    print("Warning: opacus not installed. DPSGD defense will be unavailable.")
    OPACUS_AVAILABLE = False

###############################################################################
# Magnitude Pruning (MP) ------------------------------------------------------
###############################################################################
class MPDefense:
    """模型剪枝防御"""
    def __init__(self, model, amount=0.2):
        self.model = model
        self.amount = amount
        self.feature_history = []
        self.max_history = 1000
        self.layer_stats = {}  # 存储每层的统计信息
        self._prune_model()
    
    def _calculate_layer_stats(self):
        """计算每层的权重统计信息"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                abs_weight = torch.abs(weight)
                
                # 计算每层的统计信息
                self.layer_stats[name] = {
                    'mean': abs_weight.mean().item(),
                    'std': abs_weight.std().item(),
                    'max': abs_weight.max().item(),
                    'min': abs_weight.min().item()
                }
    
    def _dynamic_threshold(self, layer_name):
        """计算动态剪枝阈值"""
        if layer_name in self.layer_stats:
            stats = self.layer_stats[layer_name]
            # 基于统计信息动态调整阈值
            base_threshold = stats['mean'] + self.amount * stats['std']
            return min(base_threshold, stats['max'] * 0.8)
        return None
    
    def _prune_model(self):
        """对模型进行增强的剪枝"""
        self._calculate_layer_stats()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 获取权重
                weight = module.weight.data
                abs_weight = torch.abs(weight)
                
                # 获取动态阈值
                threshold = self._dynamic_threshold(name)
                if threshold is None:
                    threshold = torch.quantile(abs_weight, self.amount)
                
                # 创建掩码
                mask = abs_weight > threshold
                
                # 应用掩码
                module.weight.data *= mask
                
                # 添加小的随机噪声以增强防御
                if self.model.training:
                    noise = torch.randn_like(weight) * 0.01
                    module.weight.data += noise * mask
    
    def _update_feature_history(self, features):
        """更新特征历史"""
        if self.model.training:
            self.feature_history.append(features.detach().mean(dim=0))
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
    
    def _mp_forward_hook(self, features):
        """增强的MP前向传播钩子"""
        if self.model.training:
            # 更新特征历史
            self._update_feature_history(features)
            
            # 计算特征统计信息
            if len(self.feature_history) > 0:
                feature_mean = torch.stack(self.feature_history).mean(dim=0)
                feature_std = torch.stack(self.feature_history).std(dim=0)
                
                # 对异常特征进行额外的剪枝
                anomaly_scores = torch.abs(features - feature_mean) / (feature_std + 1e-6)
                threshold = torch.quantile(anomaly_scores, 1 - self.amount)
                mask = anomaly_scores < threshold
                
                # 应用掩码
                features = features * mask.float()
                
                # 添加小的随机噪声
                noise = torch.randn_like(features) * 0.01
                features = features + noise * mask.float()
        
        return features
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self._mp_forward_hook, instance=self)

###############################################################################
# Activation Noise Perturbation (ANP) ----------------------------------------
###############################################################################
class _GaussianNoise(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x

class ANPDefense:
    """自适应噪声扰动防御"""
    def __init__(self, model, sigma=0.1):
        self.model = model
        self.sigma = sigma
        self.feature_history = []
        self.max_history = 1000
    
    def _anp_forward_hook(self, features):
        """增强的ANP前向传播钩子"""
        if self.model.training:
            # 更新特征历史
            self.feature_history.append(features.detach().mean(dim=0))
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
            
            # 计算特征统计信息
            if len(self.feature_history) > 0:
                feature_mean = torch.stack(self.feature_history).mean(dim=0)
                feature_std = torch.stack(self.feature_history).std(dim=0)
                
                # 计算每个样本与均值的距离
                distances = torch.abs(features - feature_mean)
                
                # 对异常值应用更强的噪声
                noise = torch.randn_like(features)
                noise_scale = self.sigma * (1 + distances / (feature_std + 1e-6))
                features = features + noise * noise_scale.unsqueeze(1)
            
            # 裁剪特征值范围
            features = torch.clamp(features, -10, 10)
        
        return features
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self._anp_forward_hook, instance=self)

###############################################################################
# Backdoor Detection & Trimming (BDT) ----------------------------------------
###############################################################################
class BDTDefense:
    """后门检测和修剪防御"""
    def __init__(self, model, prune_ratio=0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        self.feature_history = []
        self.max_history = 1000
        self.is_analyzed = False  # 添加分析状态标志
    
    def _bdt_forward_hook(self, features):
        """BDT前向传播钩子"""
        if not self.is_analyzed:
            return features  # 如果未分析，直接返回原始特征
            
        if self.model.training:
            # 更新特征历史
            self.feature_history.append(features.detach())
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
            
            # 计算特征统计信息
            if len(self.feature_history) > 0:
                feature_mean = torch.stack(self.feature_history).mean(dim=0)
                feature_std = torch.stack(self.feature_history).std(dim=0)
                
                # 计算每个特征的异常分数
                anomaly_scores = torch.abs(features - feature_mean) / (feature_std + 1e-6)
                
                # 找出异常特征
                threshold = torch.quantile(anomaly_scores, 1 - self.prune_ratio)
                mask = anomaly_scores < threshold
                
                # 应用掩码并添加额外的噪声
                features = features * mask
                
                # 添加额外的噪声来增强防御
                noise = torch.randn_like(features) * 0.1
                features = features + noise
        
        return features
    
    def run_bdt_offline(self, activations):
        """运行离线分析"""
        self.feature_history = activations
        self.is_analyzed = True
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self._bdt_forward_hook, instance=self)

###############################################################################
# VFLIP -----------------------------------------------------------------------
###############################################################################
class _SimpleMAE(nn.Module):
    """Minimal MLP AutoEncoder for feature reconstruction."""
    def __init__(self, input_dim: int, hid: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, input_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

class VFLIPDefense:
    """特征空间异常检测防御"""
    def __init__(self, input_dim, threshold=3.0, device="cpu"):
        self.mae = _SimpleMAE(input_dim).to(device)
        self.loss_fn = nn.MSELoss()
        self.opt = optim.Adam(self.mae.parameters(), lr=1e-3)
        self.threshold = threshold
        self.device = device
        self.feature_history = []
        self.max_history = 1000
        self.is_trained = False  # 添加训练状态标志
    
    def train_step(self, features):
        """训练自编码器"""
        self.mae.train()
        features = features.to(self.device)
        
        # 更新特征历史
        self.feature_history.append(features.detach())
        if len(self.feature_history) > self.max_history:
            self.feature_history.pop(0)
        
        # 训练自编码器
        recon = self.mae(features)
        loss = self.loss_fn(recon, features)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.is_trained = True  # 标记为已训练
        return loss.item()
    
    def _vflip_forward_hook(self, features):
        """VFLIP前向传播钩子"""
        if not self.is_trained:
            return features  # 如果未训练，直接返回原始特征
            
        self.mae.eval()
        features = features.to(self.device)
        
        # 计算重构误差
        with torch.no_grad():
            recon = self.mae(features)
            errors = (features - recon).pow(2).mean(dim=1)
        
        # 计算动态阈值
        if len(self.feature_history) > 0:
            feature_mean = torch.stack(self.feature_history).mean(dim=0)
            feature_std = torch.stack(self.feature_history).std(dim=0)
            dynamic_threshold = self.threshold * (1 + feature_std.mean())
        else:
            dynamic_threshold = self.threshold
        
        # 应用掩码并添加额外的噪声
        mask = (errors < dynamic_threshold).float().unsqueeze(1)
        features = features * mask
        
        # 添加额外的噪声来增强防御
        if self.mae.training:
            noise = torch.randn_like(features) * 0.1
            features = features + noise
        
        return features
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self._vflip_forward_hook, instance=self)


###############################################################################
# ISO (Isolation) -------------------------------------------------------------
###############################################################################
class ISODefense(nn.Module):
    """特征空间隔离层"""
    def __init__(self, input_dim, device="cpu"):
        super().__init__()
        self.iso_layer = nn.Linear(input_dim, 1).to(device)
        self.device = device
        self.feature_history = []
        self.max_history = 1000
        self.is_trained = False  # 添加训练状态标志
    
    def forward(self, x):
        """前向传播"""
        if not self.is_trained:
            return x  # 如果未训练，直接返回原始特征
            
        x = x.to(self.device)
        
        # 更新特征历史
        if self.training:
            self.feature_history.append(x.detach())
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
        
        # 计算隔离分数
        iso_score = torch.sigmoid(self.iso_layer(x))
        
        # 计算动态阈值
        if len(self.feature_history) > 0:
            feature_mean = torch.stack(self.feature_history).mean(dim=0)
            feature_std = torch.stack(self.feature_history).std(dim=0)
            dynamic_threshold = 0.5 * (1 + feature_std.mean())
        else:
            dynamic_threshold = 0.5
        
        # 应用掩码并添加额外的噪声
        mask = (iso_score > dynamic_threshold).float()
        x = x * mask
        
        # 添加额外的噪声来增强防御
        if self.training:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        
        return x
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self.forward, instance=self)


###############################################################################
# Factory ---------------------------------------------------------------------
###############################################################################
_DEF = {
    "DPSGD": None,
    "MP": MPDefense,
    "ANP": ANPDefense,
    "BDT": BDTDefense,
    "VFLIP": VFLIPDefense,
    "ISO": ISODefense,
}

class DPSGDDefense:
    """差分隐私随机梯度下降防御"""
    def __init__(self, model, optimizer, batch_size, sample_size, noise_multiplier=0.5, max_grad_norm=1.0, target_epsilon=8.0, target_delta=1e-5, train_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.train_loader = train_loader  # 新增 train_loader 参数

        # 检查 train_loader 是否为 None
        if self.train_loader is None:
            raise ValueError("train_loader cannot be None for DPSGD defense")

        # 初始化隐私引擎
        self.privacy_engine = PrivacyEngine()
        
        # 使模型私有化，在这里传入 batch_size
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        
        # 初始化RDP会计
        self.accountant = RDPAccountant()
        
        # 计算采样率
        self.sample_rate = batch_size / sample_size
        
        # 初始化隐私预算
        self.epsilon = 0
        self.delta = target_delta
        
        # 初始化梯度统计
        self.grad_norms = []
        self.max_history = 1000
    
    def _clip_gradients(self, model):
        """对每个层的梯度进行裁剪"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 更新梯度统计
        self.grad_norms.append(total_norm)
        if len(self.grad_norms) > self.max_history:
            self.grad_norms.pop(0)
        
        # 动态调整裁剪阈值
        if len(self.grad_norms) > 0:
            avg_norm = sum(self.grad_norms) / len(self.grad_norms)
            dynamic_clip = min(self.max_grad_norm, avg_norm * 1.5)
        else:
            dynamic_clip = self.max_grad_norm
        
        # 应用裁剪
        clip_coef = dynamic_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
    
    def _add_noise(self, features):
        """添加自适应噪声"""
        if self.model.training:
            # 计算特征统计信息
            feature_mean = features.mean(dim=0)
            feature_std = features.std(dim=0)
            
            # 计算动态噪声乘数
            dynamic_noise = self.noise_multiplier * (1 + feature_std.mean())
            
            # 添加噪声
            noise = torch.randn_like(features) * dynamic_noise
            features = features + noise
            
            # 更新隐私预算
            self.epsilon = self.privacy_engine.get_epsilon(self.delta)
            
            # 如果隐私预算超过目标，增加噪声
            if self.epsilon > self.target_epsilon:
                self.noise_multiplier *= 1.1
                print(f"Privacy budget exceeded. Increasing noise multiplier to {self.noise_multiplier:.2f}")
        
        return features
    
    def forward(self, features):
        """前向传播"""
        # 应用梯度裁剪
        self._clip_gradients(self.model)
        
        # 添加噪声
        features = self._add_noise(features)
        
        return features
    
    def extra_hooks(self):
        """返回额外的钩子"""
        return SimpleNamespace(forward=self.forward, instance=self)

def build_defense(model: nn.Module,
                  optimizer: optim.Optimizer | None,
                  defense_type: str = "NONE",
                  **kwargs) -> Tuple[nn.Module, optim.Optimizer | None, SimpleNamespace]:
    """构建防御机制
    
    Args:
        model: 要防御的模型
        optimizer: 模型的优化器
        defense_type: 防御类型 ('DPSGD', 'MP', 'ANP', 'BDT', 'VFLIP', 'ISO')
        **kwargs: 其他参数
    
    Returns:
        model: 应用防御后的模型
        optimizer: 更新后的优化器
        hooks: 包含前向传播钩子和其他必要组件的命名空间
    """
    hooks = SimpleNamespace(forward=None, instance=None)
    
    if defense_type.upper() == 'DPSGD':
        if optimizer is None:
            raise ValueError("DPSGD requires an optimizer")
        
        # 获取参数
        noise_multiplier = kwargs.get('noise_multiplier', 0.5)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        batch_size = kwargs.get('batch_size', 128)
        sample_size = kwargs.get('sample_size', 50000)
        target_epsilon = kwargs.get('target_epsilon', 8.0)
        target_delta = kwargs.get('target_delta', 1e-5)
        train_loader = kwargs.get('train_loader')  # 新增 train_loader 参数
        
        # 检查 train_loader 是否为 None
        if train_loader is None:
            raise ValueError("train_loader cannot be None for DPSGD defense")
        
        # 创建DPSGD防御实例
        inst = DPSGDDefense(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            sample_size=sample_size,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            train_loader=train_loader  # 传入 train_loader 参数
        )
        
        return model, optimizer, inst.extra_hooks()
    
    elif defense_type.upper() == 'MP':
        inst = MPDefense(model, amount=kwargs.get("pruning_amount", 0.2))
        return model, optimizer, inst.extra_hooks()
    
    elif defense_type.upper() == 'ANP':
        inst = ANPDefense(model, sigma=kwargs.get("sigma", 0.1))
        return model, optimizer, inst.extra_hooks()
    
    elif defense_type.upper() == 'BDT':
        inst = BDTDefense(model, prune_ratio=kwargs.get("prune_ratio", 0.2))
        return model, optimizer, inst.extra_hooks()
    
    elif defense_type.upper() == 'VFLIP':
        inst = VFLIPDefense(
            input_dim=kwargs["input_dim"],
            threshold=kwargs.get("threshold", 3.0),
            device=kwargs.get("device", "cpu")
        )
        return model, optimizer, inst.extra_hooks()
    
    elif defense_type.upper() == 'ISO':
        inst = ISODefense(
            input_dim=kwargs["input_dim"],
            device=kwargs.get("device", "cpu")
        )
        return model, optimizer, inst.extra_hooks()
    
    else:
        # 对于 "NONE" 或未知的防御类型，返回空的 hooks
        return model, optimizer, hooks 
    
import torch
import numpy as np
from scipy.fft import dct, idct

# --- SciPy DCT/IDCT 包装：保持 torch.Tensor 接口 ---
def dct_scipy(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, D] torch tensor (on CPU/GPU)
    return: [B, D] torch tensor (same dtype/device as x)
    """
    dev, dtype = x.device, x.dtype
    X_np = dct(x.detach().cpu().numpy(), type=2, norm='ortho', axis=-1)  # CPU numpy
    return torch.from_numpy(X_np).to(dev, dtype=dtype)

def idct_scipy(X: torch.Tensor) -> torch.Tensor:
    dev, dtype = X.device, X.dtype
    x_np = idct(X.detach().cpu().numpy(), type=2, norm='ortho', axis=-1)  # CPU numpy
    return torch.from_numpy(x_np).to(dev, dtype=dtype)

def remove_by_index(tensor: torch.Tensor, removed_idx: torch.Tensor):
    if removed_idx.numel() == 0:
        return tensor  # 没有要删除的
    mask = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)
    mask[removed_idx] = False
    return tensor[mask]

@torch.no_grad()
def dct_trigger_filter(
    x: torch.Tensor,
    tau: float = 4.0,        # 频域鲁棒 z-score 阈值；越小越敏感
    k_min: int = 3,          # 判为触发所需异常频点最小个数
    keep_lowpass: int = 4,   # 忽略最前低频系数个数
    eps: float = 1e-6,
):
    """
    x: [B, 256]
    return:
      clean: [B_clean, 256]
      kept_idx: [B_clean] long
      removed_idx: [B_removed] long
      poison_mask: [B] bool  (True 表示含 trigger，被删除)
    """
    assert x.dim() == 2 and x.shape[1] == 256, "期待输入 [B,256]"
    B, D = x.shape

    # 1) SciPy DCT-II (ortho)
    X = dct_scipy(x)  # [B,256]

    # 2) median + MAD（逐样本鲁棒统计）
    median = X.median(dim=-1, keepdim=True).values                       # [B,1]
    mad = (X - median).abs().median(dim=-1, keepdim=True).values         # [B,1]
    z = (X - median).abs() / (mad + eps)                                  # [B,256]

    # 3) 异常频点掩码；保护极低频
    anomaly = z > tau                                                     # [B,256]
    if keep_lowpass > 0:
        lp = min(keep_lowpass, D)
        anomaly[:, :lp] = False

    # 4) 判定整条 embedding 是否“含触发”
    count_anom = anomaly.sum(dim=-1)                                      # [B]
    poison_mask = count_anom >= k_min                                     # [B] True=删
    print(f"[My Defense] poison_mask: {poison_mask}")

    # 5) 过滤
    kept_idx = (~poison_mask).nonzero(as_tuple=False).squeeze(-1)
    removed_idx = (poison_mask).nonzero(as_tuple=False).squeeze(-1)

    clean = x[kept_idx] if kept_idx.numel() > 0 else x.new_zeros((0, D))
    return clean, kept_idx, removed_idx, poison_mask