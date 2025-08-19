#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: train_imagenette_badvfl_with_inference.py
# 描述: 针对Imagenette数据集的BadVFL攻击训练 (带标签推断)
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import torch.nn.init as init
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import time
import random
from tqdm import tqdm
from contextlib import nullcontext
import math
# === AMP混合精度 ===
import torch.cuda.amp
# === 下载相关导入 ===
import urllib.request
import tarfile
import shutil

# Add defense module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from defense_all import build_defense
except ImportError:
    print("Could not import defense_all, defense functionalities will be unavailable.")
    def build_defense(args, **kwargs):
        return {
            'pre_backward_hook': None,
            'post_backward_hook': None,
            'optimizer_step_hook': None,
        }

# 设置命令行参数
parser = argparse.ArgumentParser(description='针对Imagenette数据集的BadVFL攻击训练（带标签推断）')
parser.add_argument('--batch-size', type=int, default=64, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.95, help='动量')
parser.add_argument('--weight-decay', type=float, default=0.0008, help='权重衰减')
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--trigger-size', type=int, default=5, help='BadVFL触发器大小')
parser.add_argument('--trigger-intensity', type=float, default=0.5, help='BadVFL触发器强度')
parser.add_argument('--position', type=str, default='dr', help='触发器位置 (dr=右下, ul=左上, mid=中间, ml=中左)')
parser.add_argument('--target-class', type=int, default=0, help='目标类别')
parser.add_argument('--bkd-adversary', type=int, default=1, help='恶意方ID')
parser.add_argument('--party-num', type=int, default=4, help='参与方数量')
parser.add_argument('--patience', type=int, default=20, help='早停轮数')
parser.add_argument('--min-epochs', type=int, default=30, help='最小训练轮数')
parser.add_argument('--max-epochs', type=int, default=300, help='最大训练轮数')
parser.add_argument('--backdoor-weight', type=float, default=0.05, help='后门损失权重 (建议0.05~0.1，过高会严重影响clean acc)')
parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪')
parser.add_argument('--has-label-knowledge', type=bool, default=True, help='是否有标签知识')
parser.add_argument('--half', type=bool, default=False, help='是否使用半精度')
parser.add_argument('--log-interval', type=int, default=5, help='日志间隔')
parser.add_argument('--poison-budget', type=float, default=0.03, help='毒化预算')
parser.add_argument('--Ebkd', type=int, default=5, help='后门注入开始轮数')
parser.add_argument('--lr-multiplier', type=float, default=1.0, help='学习率倍增器')
parser.add_argument('--defense-type', type=str, default='NONE', help='防御类型')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_imagenette', help='检查点目录')
parser.add_argument('--active', type=str, default='label-knowledge', help='标签知识')
parser.add_argument('--num-classes', type=int, default=10, help='类别数量')
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data-dir', type=str, default='/home/steve/VFLIP-esorics24-master/data/Imagenette/', help='数据集目录')
parser.add_argument('--trigger-type', type=str, default='pattern', help='触发器类型 (pattern或pixel)')

# Defense arguments
parser.add_argument('--dpsgd-sigma', type=float, default=0.01, help='Sigma for DPSGD')
parser.add_argument('--dpsgd-max-grad-norm', type=float, default=1.0, help='Max grad norm for DPSGD')
parser.add_argument('--anp-alpha', type=float, default=0.05, help='Alpha for ANP')
parser.add_argument('--anp-percentile', type=int, default=10, help='Percentile for ANP')
parser.add_argument('--bdt-T', type=float, default=0.5, help='T for BDT')
parser.add_argument('--bdt-alpha', type=float, default=0.5, help='Alpha for BDT')
parser.add_argument('--vflip-beta', type=float, default=0.1, help='Beta for VFLIP')
parser.add_argument('--vflip-gamma', type=float, default=0.1, help='Gamma for VFLIP')
parser.add_argument('--vflip-delta', type=float, default=0.1, help='Delta for VFLIP')
parser.add_argument('--iso-p', type=float, default=0.5, help='p for ISO')
parser.add_argument('--iso-alpha', type=float, default=0.05, help='alpha for ISO')
parser.add_argument('--iso-beta', type=float, default=0.9, help='beta for ISO')

# 标签推断相关参数
parser.add_argument('--inference-weight', type=float, default=0.05, help='标签推断损失权重')
parser.add_argument('--history-size', type=int, default=5000, help='嵌入向量历史记录大小')
parser.add_argument('--cluster-update-freq', type=int, default=50, help='聚类更新频率(批次)')
parser.add_argument('--inference-start-epoch', type=int, default=2, help='开始标签推断的轮数')  # 更早开始标签推断
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
parser.add_argument('--monitor', type=str, default='inference_acc', choices=['test_acc', 'inference_acc'],
                    help='监控指标，用于早停判断 (default: inference_acc)')

# 源类别参数 
parser.add_argument('--source-class', type=int, default=6, help='攻击源类别')

# 设置全局变量
args = parser.parse_args()

# 参数后处理
if args.device != f"cuda:{args.gpu}" and torch.cuda.is_available():
    args.device = f"cuda:{args.gpu}"

# 全局设备变量
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

# 添加GPU内存监控函数
def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        return "无GPU可用"
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total_memory - allocated
    
    return f"GPU内存: 总计 {total_memory:.2f}GB, 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 可用 {free:.2f}GB"

def check_cuda():
    """检查CUDA是否可用，打印详细信息"""
    print("\n检查CUDA可用性...")
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用! 训练可能会非常慢。")
        return False
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, 内存: {props.total_memory / (1024**3):.2f} GB")
        
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"当前CUDA设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 尝试进行简单的CUDA操作，检查是否正常工作
    try:
        a = torch.tensor([1., 2.], device="cuda")
        b = torch.tensor([3., 4.], device="cuda")
        c = a + b
        print("CUDA测试操作成功!")
        return True
    except Exception as e:
        print(f"CUDA测试操作失败: {str(e)}")
        return False

def setup_seed(seed):
    """设置随机种子，确保实验可重复"""
    print(f"设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    global parser
    
    # 不添加任何重复的参数，只提供新的或需要覆盖默认值的参数
    # 所有已经在初始parser中定义的参数都不要再定义
    
    # 测试相关参数
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                      help='每批测试样本数 (默认: 128)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='学习率调度器步长 (默认: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='禁用CUDA训练')
    parser.add_argument('--dry-run', action='store_true', default=False,
                      help='快速干运行')
    parser.add_argument('--save-model', action='store_true', default=True,
                      help='保存当前最佳模型')
    
    # 触发器位置参数 - 不同于已有的 position 参数
    parser.add_argument('--trigger-position', type=int, default=1,
                      help='触发器位置 (默认: 1)')
    
    args = parser.parse_args()
    return args

# 定义全局变量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()

# BadVFL触发器实现
class BadVFLTrigger:
    """BadVFL攻击中的触发器实现，支持多种类型的触发器模式"""
    def __init__(self, args):
        self.args = args
        self.target_class = args.target_class
        self.device = args.device if hasattr(args, 'device') else 'cpu'
        self.dataset_name = 'Imagenette'
        self.position = args.position  # 触发器位置
        
        # 处理参数名称
        if hasattr(args, 'trigger_size'):
            self.pattern_size = int(args.trigger_size)  # 触发器大小
        else:
            self.pattern_size = 4  # 默认值
            print(f"触发器初始化: 未找到大小参数，使用默认值 pattern_size = {self.pattern_size}")
        
        # 降低触发器强度，使其不足以单独决定模型输出
        # 原来是 args.trigger_intensity * 3.0
        self.intensity = args.trigger_intensity * 0.7  # 大幅降低触发器强度
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
        # 调整为Imagenette的尺寸 - 224x224
        self.pixel_positions = [
            (0, 0), (0, 1), (1, 0),  # 左上角形成小方块
            (223, 223), (222, 223), (223, 222),  # 右下角形成小方块
            (0, 223), (1, 223),  # 左下角
            (223, 0), (223, 1),  # 右上角
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
        """创建角落模式触发器 - 极高对比度版本 - 适用于224x224图像"""
        size = self.pattern_size
        self.pattern_mask = torch.zeros(3, 224, 224)
        
        if position == 'dr':  # 右下角
            x_start, y_start = 224 - size, 224 - size
        elif position == 'ul':  # 左上角
            x_start, y_start = 0, 0
        elif position == 'ml':  # 中左
            x_start, y_start = 112 - size // 2, 0
        else:  # 默认右下角
            x_start, y_start = 224 - size, 224 - size
        
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
    
    def create_center_pattern(self):
        """创建中心模式触发器 - 适用于224x224图像"""
        size = self.pattern_size
        self.pattern_mask = torch.zeros(3, 224, 224)
        
        # 计算中心位置
        x_start = 112 - size // 2
        y_start = 112 - size // 2
        
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
            
    def inject_trigger(self, data, attack_flags=None, confidence=None):
        """向输入数据中注入触发器
        
        Args:
            data: 输入数据，形状为 [batch_size, channels, height, width]
            attack_flags: 指示哪些样本应该被攻击的布尔掩码，可以是Tensor或列表
            confidence: 推断置信度，用于动态调整触发器强度
            
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
        
        # 定义基础触发器强度
        base_intensity = self.intensity
        
        # 如果提供了置信度，则进行动态强度调整
        if confidence is not None and isinstance(confidence, (np.ndarray, list)) and len(confidence) == len(data):
            # 根据置信度调整触发器强度
            # 当置信度接近0.5时触发器较弱，越接近0或1触发器越强
            # 这确保了触发器强度与推断准确度直接相关
            adjusted_intensities = []
            for i in range(len(data)):
                if not attack_flags[i]:
                    adjusted_intensities.append(0.0)  # 不攻击的样本强度为0
                    continue
                    
                # 获取当前样本的置信度
                conf = confidence[i]
                
                # 计算置信度偏离0.5的程度（0.5表示最不确定）
                conf_deviation = abs(conf - 0.5) * 2  # 范围[0,1]
                
                # 根据置信度调整强度：置信度越高，强度越低
                # 当置信度为0.5时，使用最低强度 (base_intensity * 0.3)
                # 当置信度为0或1时，使用最高强度 (base_intensity * 1.0)
                adjusted_intensity = base_intensity * (0.3 + 0.7 * conf_deviation)
                adjusted_intensities.append(adjusted_intensity)
                
            # 转换为张量
            if not isinstance(adjusted_intensities, torch.Tensor):
                adjusted_intensities = torch.tensor(adjusted_intensities, device=device)
        else:
            # 如果没有置信度信息，使用基础强度
            adjusted_intensities = torch.ones(len(data), device=device) * base_intensity
        
        # 打印强度统计信息
        if attack_flags.sum() > 0:
            attack_intensities = adjusted_intensities[attack_flags]
            avg_intensity = attack_intensities.mean().item()
            min_intensity = attack_intensities.min().item()
            max_intensity = attack_intensities.max().item()
            print(f"触发器强度统计: 平均={avg_intensity:.4f}, 最小={min_intensity:.4f}, 最大={max_intensity:.4f}")
        
        # 注入触发器 - 使用更小的触发器尺寸和更自然的混合方式
        if self.trigger_type == 'pixel':
            # 更小的像素块大小，从2x2降低到1x1单个像素
            pixel_block_size = 1
            
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 获取当前样本的动态强度
                    current_intensity = adjusted_intensities[idx]
                    
                    # 对被攻击样本应用触发器
                    temp_data = data_copy[idx].clone()
                    
                    # 只使用前3个触发点，进一步减少影响区域
                    selected_positions = self.pixel_positions[:3]
                    selected_values = self.pixel_values[:3]
                    
                    for (x, y), (r, g, b) in zip(selected_positions, selected_values):
                        # 使用非常低的透明度，让触发器几乎不可见
                        alpha = 0.3  # 降低透明度
                        temp_data[0, x, y] = (1-alpha) * temp_data[0, x, y] + alpha * r * current_intensity
                        temp_data[1, x, y] = (1-alpha) * temp_data[1, x, y] + alpha * g * current_intensity
                        temp_data[2, x, y] = (1-alpha) * temp_data[2, x, y] + alpha * b * current_intensity
                    
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
            
            # 为每个需要攻击的样本应用小触发器
            for idx in range(len(data)):
                if attack_flags[idx]:
                    # 获取当前样本的动态强度
                    current_intensity = adjusted_intensities[idx]
                    
                    # 创建掩码，指示哪些像素需要修改
                    mask = (pattern_mask > 0).float()
                    
                    # 创建临时数据以避免修改原始数据
                    temp_data = data_copy[idx].clone()
                    
                    # 使用动态强度和低透明度
                    alpha = 0.25  # 非常低的透明度
                    
                    # 使用混合函数，避免完全替换原像素
                    blended = (1 - mask * alpha) * temp_data + pattern_mask * current_intensity * alpha
                    
                    # 确保值在有效范围内
                    blended = torch.clamp(blended, 0.0, 1.0)
                    
                    # 将处理后的数据复制回data_copy
                    data_copy[idx] = blended
        
        return data_copy
    
    def inject_trigger_with_inference(self, data, attack_flags=None, raw_data=None, top_model=None, bottom_models=None):
        """基于标签推断选择性地注入触发器 - 修复版本，更好地处理标签推断结果
        
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
        print("使用标签推断选择攻击样本...")
        inferred_labels, confidences = self.label_inference.infer_labels(
            data.view(batch_size, -1), top_model, bottom_models, raw_data
        )
        
        if inferred_labels is None:
            # 推断失败，放弃攻击
            print("警告: 标签推断失败，无法进行攻击")
            # 返回原始数据，不注入触发器
            return data
        
        # 创建基于推断的攻击标志
        # 只有推断为非目标类(0)的样本才会被攻击
        # 这确保攻击完全依赖于标签推断
        inference_attack_flags = torch.zeros(batch_size, dtype=torch.bool, device=data.device)
        
        # 统计标签分布
        target_class_count = 0
        non_target_class_count = 0
        
        # 仅根据推断结果决定攻击哪些样本，完全忽略原始attack_flags
        for i in range(batch_size):
            if inferred_labels[i] == 1:  # 推断为目标类
                target_class_count += 1
                # 不攻击目标类样本，无论原始attack_flags如何
            else:  # 推断为非目标类
                non_target_class_count += 1
                # 只攻击推断为非目标类的样本
                inference_attack_flags[i] = True
        
        # 打印详细的标签推断结果
        print(f"标签推断结果: 目标类={target_class_count}, 非目标类={non_target_class_count}")
        
        # 计算与原始攻击标志的一致性（仅用于调试，不影响决策）
        if attack_flags is not None:
            original_attack_count = attack_flags.sum().item()
            inference_attack_count = inference_attack_flags.sum().item()
            
            # 计算有多少攻击标志与推断结果一致
            consistent_count = 0
            for i in range(batch_size):
                if attack_flags[i] and inference_attack_flags[i]:
                    consistent_count += 1
            
            consistency_ratio = consistent_count / original_attack_count if original_attack_count > 0 else 0
            print(f"攻击标志比较: 原始攻击={original_attack_count}, 基于推断攻击={inference_attack_count}")
            print(f"一致性: {consistent_count}/{original_attack_count} = {consistency_ratio:.2f}")
            
            # 如果一致性太低，发出警告
            if consistency_ratio < 0.5:
                print("警告: 标签推断与原始攻击标志一致性很低，可能影响ASR")
        
        # 如果标签推断没有识别任何非目标类样本，放弃攻击
        if inference_attack_flags.sum().item() == 0:
            print("警告: 标签推断未识别任何非目标类样本，无法进行攻击")
            # 返回原始数据，不注入触发器
            return data
        
        # 使用推断结果选择攻击样本，完全依赖标签推断
        return self.inject_trigger(data, inference_attack_flags)

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
        
        # GPU版缓存 - 初始化为空张量而非None，避免torch.cat错误
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
        
        # 设置所需的最小样本数 - 大幅降低所需的最小样本数以加快初始化
        self.min_samples = 20  # 降低到只需要20个样本即可初始化
        
        # 创建GPU分类器 - PyTorch神经网络实现
        self._create_gpu_classifier(input_dim=feature_dim)
        print(f"BadVFL标签推断模块创建: 特征维度={feature_dim}, 类别数={num_classes}")
        print(f"标签推断所需最小样本数: {self.min_samples}")
        print(f"已创建GPU版分类器，将使用设备: {DEVICE}")
        
        # 确保初始化后不会出现None的情况
        self._initialize_empty_tensors()
    
    def _create_gpu_classifier(self, input_dim=None):
        """创建GPU版分类器 - 使用更简单但更有效的网络结构以提高标签推断准确率"""
        class ImprovedClassifier(nn.Module):
            def __init__(self, input_dim):
                super(ImprovedClassifier, self).__init__()
                
                # 更简单但更有效的网络结构
                self.fc1 = nn.Linear(input_dim, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.dropout2 = nn.Dropout(0.4)
                
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.dropout3 = nn.Dropout(0.3)
                
                self.out = nn.Linear(128, 1)
                
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
                x = F.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                
                x = F.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                
                x = F.relu(self.bn3(self.fc3(x)))
                x = self.dropout3(x)
                
                x = self.out(x)
                return x

        # 使用传入的input_dim或self.feature_dim
        if input_dim is None:
            input_dim = self.feature_dim
            
        self.inference_classifier_gpu = ImprovedClassifier(input_dim).to(DEVICE)
        print(f"创建改进的标签推断分类器，输入维度={input_dim}，共 {sum(p.numel() for p in self.inference_classifier_gpu.parameters())} 个参数")
    
    def _initialize_empty_tensors(self):
        """确保GPU张量被正确初始化为空张量而非None"""
        if self.history_features_gpu is None:
            self.history_features_gpu = torch.FloatTensor().to(DEVICE)
            print("初始化空的特征张量")
        if self.history_predictions_gpu is None:
            self.history_predictions_gpu = torch.FloatTensor().to(DEVICE)
            print("初始化空的预测张量")
        if self.history_confidence_gpu is None:
            self.history_confidence_gpu = torch.FloatTensor().to(DEVICE)
            print("初始化空的置信度张量")
    
    def update_with_batch(self, features, predictions, confidence=None):
        """更新特征和预测历史记录，处理梯度数据以提取类别信息"""
        # 确保GPU张量已初始化
        self._initialize_empty_tensors()
        
        # 保存原始数据类型
        is_tensor = isinstance(features, torch.Tensor)
        
        # 为CPU版处理准备数据
        features_cpu = features.detach().cpu().numpy() if is_tensor else features
        predictions_cpu = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        
        # 打印输入数据的形状和类型，以便调试
        print(f"更新标签推断记录 - 特征形状: {features_cpu.shape}, 预测/梯度形状: {predictions_cpu.shape}, 类型: {predictions_cpu.dtype}")
        
        # 我们处理的是梯度信息，不是真实的类别标签
        # 直接记录样本预测为目标类的概率(1)或非目标类的概率(0)
        # 正确处理梯度信息：对于损失函数，梯度通常包含丰富的类别信息
        
        # 处理方法：分析梯度与目标类和非目标类的相关性
        # 使用一个更简单但更可靠的方法
        if np.issubdtype(predictions_cpu.dtype, np.floating):  # 如果是浮点类型，说明是梯度
            # 简化的二分类标签生成 - 将数据分为两类
            # 主要思想：目标类和非目标类在损失梯度上应该有区别
            # 我们使用梯度的绝对值大小作为区分依据
            
            # 1. 展平梯度，若为多维
            if len(predictions_cpu.shape) > 1:
                # 计算每个样本梯度的L2范数（强度）
                grad_magnitudes = np.linalg.norm(predictions_cpu, axis=1)
            else:
                grad_magnitudes = np.abs(predictions_cpu)
                
            # 2. 使用阈值进行二分类
            # 这里我们使用中位数作为阈值，大于中位数的认为是目标类(1)，小于的认为是非目标类(0)
            threshold = np.median(grad_magnitudes)
            
            # 3. 生成二分类标签
            binary_predictions = (grad_magnitudes > threshold).astype(np.int32)
            
            # 打印标签分布
            pos_count = np.sum(binary_predictions == 1)
            neg_count = np.sum(binary_predictions == 0)
            print(f"从梯度生成的标签分布 - 标签1(可能是目标类): {pos_count}, 标签0(可能是非目标类): {neg_count}")
            
            predictions_cpu = binary_predictions
        
        # 如果没有提供置信度，使用默认值
        if confidence is None:
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
        
        # 创建预测和置信度张量
        predictions_tensor = torch.tensor(predictions_cpu, dtype=torch.float32, device=DEVICE)
        confidence_tensor = torch.tensor(confidence_cpu, dtype=torch.float32, device=DEVICE)
        
        # 创建或更新GPU历史记录
        if self.history_features_gpu is None:
            # 初次创建GPU历史记录
            self.history_features_gpu = features
            self.history_predictions_gpu = predictions_tensor
            self.history_confidence_gpu = confidence_tensor
            print(f"创建GPU历史记录，大小: {len(self.history_features_gpu)}")
        else:
            # 添加新数据到GPU历史记录
            self.history_features_gpu = torch.cat([self.history_features_gpu, features], dim=0)
            self.history_predictions_gpu = torch.cat([self.history_predictions_gpu, predictions_tensor], dim=0)
            self.history_confidence_gpu = torch.cat([self.history_confidence_gpu, confidence_tensor], dim=0)
            # 控制GPU历史记录大小
            max_gpu_history = 512
            if len(self.history_features_gpu) > max_gpu_history:
                self.history_features_gpu = self.history_features_gpu[-max_gpu_history:]
                self.history_predictions_gpu = self.history_predictions_gpu[-max_gpu_history:]
                self.history_confidence_gpu = self.history_confidence_gpu[-max_gpu_history:]
        
        print(f"更新后的历史记录大小: {len(self.history_features_gpu)}")
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
    
    def _init_pytorch_classifier(self):
        """初始化PyTorch分类器 (GPU版本) - 改进训练策略提高推断准确率"""
        # 设置固定随机种子确保结果可重现
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # 确保历史记录已转换为GPU张量
        if self.history_features_gpu is None or len(self.history_features_gpu) < self.min_samples:
            # 如果GPU历史记录不存在或样本不足，从CPU历史记录转换
            X_cpu = np.array(self.history_features_cpu[:self.args.history_size])
            y_cpu = np.array(self.history_predictions_cpu[:self.args.history_size])
            
            # 明确定义二分类标签：
            # 1 = 目标类 (target_class)
            # 0 = 非目标类 (其他所有类)
            binary_labels = (y_cpu == self.args.target_class).astype(np.float32)
            
            # 打印标签分布
            target_count = np.sum(binary_labels == 1)
            non_target_count = np.sum(binary_labels == 0)
            print(f"标签分布: 目标类(标签1)={target_count}, 非目标类(标签0)={non_target_count}")
            
            # 转换为GPU张量 - 确保指定了正确的设备
            self.history_features_gpu = torch.FloatTensor(X_cpu).to(DEVICE)
            self.history_predictions_gpu = torch.FloatTensor(binary_labels).to(DEVICE)
            
            # 打印设备信息以验证
            print(f"标签推断数据已加载到: {self.history_features_gpu.device}")
        
        # 确保神经网络在正确的设备上
        self.inference_classifier_gpu = self.inference_classifier_gpu.to(DEVICE)
        print(f"标签推断分类器在设备: {next(self.inference_classifier_gpu.parameters()).device}")
        
        try:
            # 训练分类器 - 使用改进的训练策略
            print(f"训练标签推断分类器，使用 {len(self.history_features_gpu)} 个样本...")
            self.inference_classifier_gpu.train()
            features = self.history_features_gpu
            labels = self.history_predictions_gpu
            
            if features.device != DEVICE:
                features = features.to(DEVICE)
            if labels.device != DEVICE:
                labels = labels.to(DEVICE)
                
            positive_count = labels.sum().item()
            total_count = len(labels)
            negative_count = total_count - positive_count
            pos_ratio = positive_count / total_count if total_count > 0 else 0
            print(f"标签推断训练集分布: 目标类(标签1)={positive_count}({pos_ratio:.1%}), 非目标类(标签0)={negative_count}({1-pos_ratio:.1%})")
            
            # 数据平衡处理 - 改进版本
            if positive_count > 0 and negative_count > 0:
                # 计算平衡后的数据集大小 - 确保是整数
                target_size = int(min(positive_count, negative_count) * 2)  # 每类最多取较少类的2倍
                
                # 获取目标类和非目标类的索引
                target_indices = torch.where(labels == 1)[0]
                non_target_indices = torch.where(labels == 0)[0]
                
                # 平衡采样 - 确保所有索引都是整数
                target_samples_needed = int(target_size // 2)
                non_target_samples_needed = int(target_size // 2)
                
                if len(target_indices) > target_samples_needed:
                    selected_target = target_indices[torch.randperm(len(target_indices))[:target_samples_needed]]
                else:
                    selected_target = target_indices
                    
                if len(non_target_indices) > non_target_samples_needed:
                    selected_non_target = non_target_indices[torch.randperm(len(non_target_indices))[:non_target_samples_needed]]
                else:
                    selected_non_target = non_target_indices
                
                # 合并平衡后的数据
                balanced_indices = torch.cat([selected_target, selected_non_target])
                balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]  # 打乱顺序
                
                features = features[balanced_indices]
                labels = labels[balanced_indices]
                
                print(f"数据平衡后: 总样本={len(features)}, 目标类={(labels==1).sum().item()}, 非目标类={(labels==0).sum().item()}")
            
            # 优化训练参数
            epochs = 80  # 减少训练轮数防止过拟合
            batch_size = min(32, max(8, len(features) // 4))  # 更小的批次
            
            # 使用AdamW优化器，更好的泛化性能
            optimizer = torch.optim.AdamW(
                self.inference_classifier_gpu.parameters(), 
                lr=0.001,  # 更合适的学习率
                weight_decay=1e-4,  # 轻微正则化
                betas=(0.9, 0.999)
            )
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            # 使用标准的加权BCE损失
            pos_weight = torch.tensor([negative_count / positive_count]).to(DEVICE) if positive_count > 0 else torch.tensor([1.0]).to(DEVICE)
            pos_weight = torch.clamp(pos_weight, 0.5, 3.0)  # 限制权重范围
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            print(f"使用加权BCE损失，正例权重: {pos_weight.item():.2f}")
            
            # 训练循环
            epoch_iterator = tqdm(range(epochs), desc="训练标签推断分类器")
            best_accuracy = 0
            patience_counter = 0
            early_stop_patience = 15
            
            for epoch in epoch_iterator:
                num_batches = (len(features) + batch_size - 1) // batch_size
                indices = torch.randperm(len(features), device=DEVICE)
                total_loss = 0
                correct = 0
                total = 0
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(features))
                    batch_indices = indices[start_idx:end_idx]
                    batch_features = features[batch_indices]
                    batch_labels = labels[batch_indices]
                    
                    optimizer.zero_grad()
                    outputs = self.inference_classifier_gpu(batch_features).squeeze()
                    if len(batch_indices) == 1:
                        outputs = outputs.unsqueeze(0)
                    
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.inference_classifier_gpu.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                scheduler.step()
                
                epoch_accuracy = 100 * correct / total if total > 0 else 0
                epoch_iterator.set_postfix(loss=total_loss/num_batches, acc=f"{epoch_accuracy:.2f}%", lr=f"{scheduler.get_last_lr()[0]:.6f}")
                
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= early_stop_patience and epoch > 20:
                    print(f"早停: {early_stop_patience} 轮无改善")
                    break
            
            print(f"标签推断训练完成，最佳准确率: {best_accuracy:.2f}%")
            
            # 最终验证
            self.inference_classifier_gpu.eval()
            with torch.no_grad():
                all_outputs = self.inference_classifier_gpu(features).squeeze()
                if len(features) == 1:
                    all_outputs = all_outputs.unsqueeze(0)
                predicted = (torch.sigmoid(all_outputs) > 0.5).float()
                final_accuracy = (predicted == labels).float().mean().item() * 100
                
                # 计算详细指标
                true_pos = ((predicted == 1) & (labels == 1)).sum().item()
                true_neg = ((predicted == 0) & (labels == 0)).sum().item()
                false_pos = ((predicted == 1) & (labels == 0)).sum().item()
                false_neg = ((predicted == 0) & (labels == 1)).sum().item()
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"最终验证结果: 准确率={final_accuracy:.2f}%, 精确率={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")
                print(f"混淆矩阵: TP={true_pos}, TN={true_neg}, FP={false_pos}, FN={false_neg}")
            
            return True
            
        except Exception as e:
            print(f"GPU标签推断训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            return False
    
    def infer_labels(self, features, top_model=None, bottom_models=None, raw_data=None):
        self._initialize_empty_tensors()
        # 自动适配特征维度
        if isinstance(features, torch.Tensor):
            feature_dim = features.shape[1]
        else:
            feature_dim = features.shape[1] if hasattr(features, 'shape') else len(features[0])
        if feature_dim != self.feature_dim:
            print(f"[标签推断修复] 特征维度不一致！推断时: {feature_dim}, 训练时: {self.feature_dim}")
            print("自动重建标签推断分类器并同步特征维度。\n")
            self.feature_dim = feature_dim
            self._create_gpu_classifier(input_dim=feature_dim)
            self.initialized = False
            # 重新初始化分类器参数（如果有历史数据）
            if self.history_features_gpu is not None and self.history_features_gpu.shape[1] == feature_dim:
                self.initialize_classifier()
            else:
                print("历史特征shape不匹配，需重新收集标签推断数据！")
                return None, None
        
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
                
                # 特征标准化 - 重要改进！
                # 使用训练时的统计信息进行标准化
                if hasattr(self, 'training_mean') and hasattr(self, 'training_std'):
                    features = (features - self.training_mean) / (self.training_std + 1e-8)
                else:
                    # 如果没有训练统计信息，使用当前特征的统计信息
                    feature_mean = features.mean(dim=0, keepdim=True)
                    feature_std = features.std(dim=0, keepdim=True)
                    features = (features - feature_mean) / (feature_std + 1e-8)
                
                # 设置自适应批次大小
                if torch.cuda.is_available():
                    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) - torch.cuda.memory_allocated() / (1024 * 1024)
                    estimated_mem_per_sample = 10  # MB
                    batch_size = max(32, min(512, int(free_mem / (estimated_mem_per_sample * 1.5))))
                else:
                    batch_size = 128
                
                all_logits = []
                all_confidences = []
                self.inference_classifier_gpu.eval()
                
                # 分批处理
                with torch.no_grad():
                    for i in range(0, len(features), batch_size):
                        batch_features = features[i:i + batch_size]
                        outputs = self.inference_classifier_gpu(batch_features).squeeze()
                        probs = torch.sigmoid(outputs)
                        
                        # 保存原始logits和概率用于自适应阈值
                        all_logits.append(outputs.cpu().numpy())
                        all_confidences.append(probs.cpu().numpy())
                
                # 合并结果
                if len(all_logits) > 1:
                    logits = np.concatenate(all_logits)
                    confidence = np.concatenate(all_confidences)
                else:
                    logits = all_logits[0] if len(all_logits) > 0 else np.array([])
                    confidence = all_confidences[0] if len(all_confidences) > 0 else np.array([])
                
                if len(confidence) == 0:
                    return None, None
                
                # 改进的自适应阈值策略
                threshold = 0.5  # 默认阈值
                
                # 使用更智能的阈值选择策略
                mean_confidence = np.mean(confidence)
                std_confidence = np.std(confidence)
                
                # 策略1: 基于置信度分布的自适应阈值
                if std_confidence > 0.1:  # 如果置信度分布足够分散
                    # 使用分位数作为阈值
                    q25 = np.percentile(confidence, 25)
                    q75 = np.percentile(confidence, 75)
                    
                    if q75 - q25 > 0.3:  # 四分位距足够大
                        threshold = np.percentile(confidence, 50)  # 使用中位数
                        print(f"使用中位数阈值: {threshold:.3f}")
                    else:
                        # 使用传统的平均值调整
                        if mean_confidence < 0.3:
                            threshold = 0.3
                        elif mean_confidence > 0.7:
                            threshold = 0.7
                        else:
                            threshold = 0.5
                        print(f"使用调整后阈值: {threshold:.3f} (均值: {mean_confidence:.3f})")
                else:
                    # 置信度分布太集中，使用固定阈值
                    if mean_confidence < 0.2:
                        threshold = 0.2
                    elif mean_confidence > 0.8:
                        threshold = 0.8
                    else:
                        threshold = 0.5
                    print(f"使用固定阈值: {threshold:.3f} (分布过于集中)")
                
                # 策略2: 基于logits的调整
                mean_logits = np.mean(logits)
                if abs(mean_logits) > 2:  # logits过于极端
                    print(f"检测到极端logits (均值: {mean_logits:.3f})，调整阈值策略")
                    if mean_logits > 2:  # 大多数预测为正类
                        threshold = min(0.8, threshold + 0.1)
                    else:  # 大多数预测为负类
                        threshold = max(0.2, threshold - 0.1)
                
                # 使用最终阈值进行预测
                binary_pred = (confidence > threshold).astype(float)
                
                # 计算预测统计
                pos_count = np.sum(binary_pred == 1)
                neg_count = np.sum(binary_pred == 0)
                print(f"推断结果: 目标类={pos_count}, 非目标类={neg_count}, 阈值={threshold:.3f}")
                print(f"置信度统计: 均值={mean_confidence:.3f}, 标准差={std_confidence:.3f}")
                
                return binary_pred, confidence
                
            except Exception as e:
                print(f"GPU标签推断失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
        
        print("调试 - 推断未初始化或分类器不存在")
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

# Imagenette模型类定义
class BottomModelResNet(nn.Module):
    """Imagenette底部模型，基于ResNet18架构"""
    def __init__(self, output_dim=64, party_num=4, is_adversary=False, args=None):
        super(BottomModelResNet, self).__init__()
        self.output_dim = output_dim
        self.is_adversary = is_adversary
        self.args = args
        
        # 使用预训练的ResNet18作为特征提取器，但移除最后的FC层
        self.model = models.resnet18(pretrained=True)
        # 移除最后的全连接层
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()  # 替换成恒等映射，保留特征
        
        # 添加新的输出层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
        # 冻结ResNet的大部分参数以加快训练
        # 只训练最后两层卷积块和新添加的FC层
        ct = 0
        for child in self.model.children():
            ct += 1
            # 冻结前6个模块(包括conv1, bn1, relu, maxpool, layer1, layer2)
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        
        # 用于存储当前批次的数据和梯度
        self.current_batch_data = None
        self.current_batch_grad = None
        
        # 如果是恶意模型，初始化标签推断模块
        if is_adversary and args is not None:
            self.badvfl_trigger = None
            self.label_inference = BadVFLLabelInference(
                feature_dim=self.output_dim,
                num_classes=args.num_classes,
                args=args
            )
            print(f"创建恶意底部模型 (ID={args.bkd_adversary})")
    
    def forward(self, x, attack_flags=None):
        """前向传播，包括特征提取和分类"""
        # 确保输入数据在正确的设备上
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        # 如果是恶意模型，保存输入数据用于梯度收集
        if self.is_adversary and self.training:
            self.current_batch_data = x.detach()
            x.requires_grad_(True)
        
        # 使用ResNet提取特征
        features = self.model(x)
        
        # 应用FC层获取最终输出
        output = self.fc(features)
        
        # 如果是恶意模型且在训练模式下，注册钩子以收集梯度
        if self.is_adversary and self.training and output.requires_grad:
            output.register_hook(self._gradient_hook)
        
        return output
    
    def _gradient_hook(self, grad):
        """梯度钩子函数，用于收集梯度"""
        if self.current_batch_data is not None:
            self.current_batch_grad = grad.detach().cpu()  # 只保存数值，绝不保存graph
    
    def get_saved_data(self):
        """获取保存的数据和梯度"""
        if self.current_batch_data is not None and self.current_batch_grad is not None:
            return self.current_batch_data.detach().cpu(), self.current_batch_grad  # 都detach+cpu
        return None, None
    
    def set_badvfl_trigger(self, badvfl_trigger):
        """设置BadVFL触发器"""
        if self.is_adversary:
            self.badvfl_trigger = badvfl_trigger
            # 将标签推断模块传递给触发器
            self.badvfl_trigger.set_label_inference(self.label_inference)

class ImagenetteTopModel(nn.Module):
    """Imagenette顶部模型"""
    def __init__(self, input_dim=256, num_classes=10):
        super(ImagenetteTopModel, self).__init__()
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

def download_imagenette(data_root):
    """自动下载Imagenette数据集"""
    print(f"\n{'='*50}")
    print("Imagenette数据集未找到，开始自动下载...")
    print(f"{'='*50}")
    
    # Imagenette数据集的下载链接 (320px版本，更小更快)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    
    # 创建目标目录
    os.makedirs(data_root, exist_ok=True)
    
    # 下载文件路径
    download_path = os.path.join(data_root, "imagenette2-320.tgz")
    
    print(f"下载URL: {url}")
    print(f"保存路径: {download_path}")
    print("开始下载... (大约94MB，这可能需要几分钟)")
    
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
        
        # 检查解压后的目录结构并重新组织
        extracted_dirs = [
            os.path.join(data_root, "imagenette2-320"),
            os.path.join(data_root, "imagenette2"),
            os.path.join(data_root, "imagenette")
        ]
        
        extracted_dir = None
        for dir_path in extracted_dirs:
            if os.path.exists(dir_path):
                extracted_dir = dir_path
                break
        
        if extracted_dir:
            print(f"找到解压目录: {extracted_dir}")
            
            # 将imagenette2-320子目录的内容移动到data_root
            for item in os.listdir(extracted_dir):
                source_path = os.path.join(extracted_dir, item)
                target_path = os.path.join(data_root, item)
                
                if os.path.exists(target_path):
                    if os.path.isdir(target_path):
                        shutil.rmtree(target_path)
                    else:
                        os.remove(target_path)
                
                shutil.move(source_path, target_path)
            
            # 删除空的imagenette2-320目录
            os.rmdir(extracted_dir)
            print("目录结构调整完成!")
        
        # 删除压缩文件以节省空间
        os.remove(download_path)
        print("清理完成!")
        
        # 验证下载结果
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print(f"Imagenette数据集下载并解压成功!")
            print(f"训练集: {train_dir}")
            print(f"验证集: {val_dir}")
            
            # 统计每个集合的样本数
            try:
                train_count = sum([len(files) for r, d, files in os.walk(train_dir)])
                val_count = sum([len(files) for r, d, files in os.walk(val_dir)])
                print(f"训练样本数: {train_count}")
                print(f"验证样本数: {val_count}")
            except:
                pass
            
            return True
        else:
            print("解压后目录结构不正确")
            print(f"期望的训练目录: {train_dir}")
            print(f"期望的验证目录: {val_dir}")
            return False
            
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        print("请尝试手动下载Imagenette数据集")
        print("下载地址: https://github.com/fastai/imagenette")
        return False

def load_dataset(dataset_name, data_dir, batch_size):
    """加载Imagenette数据集 - 支持自动下载"""
    print(f"\n{'='*50}")
    print(f"开始加载 {dataset_name} 数据集")
    print(f"{'='*50}")
    
    print("\n1. 准备数据预处理...")
    # Imagenette数据集预处理 - 针对224x224像素图片的预处理
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化参数
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n2. 检查Imagenette数据集路径...")
    data_root = data_dir
    
    # 检查数据集是否已存在
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    dataset_exists = os.path.exists(train_dir) and os.path.exists(val_dir)
    
    if not dataset_exists:
        print(f"训练目录未找到。期望路径: {train_dir}")
        print(f"验证目录未找到。期望路径: {val_dir}")
        
        # 尝试查找其他可能的目录结构
        alternative_paths = [
            os.path.join(data_root, 'imagenette2-320'),
            os.path.join(data_root, 'imagenette2'),
            os.path.join(data_root, 'imagenette'),
            os.path.join(data_root, 'Imagenette')
        ]
        
        found_alternative = False
        for alt_path in alternative_paths:
            alt_train = os.path.join(alt_path, 'train')
            alt_val = os.path.join(alt_path, 'val')
            if os.path.exists(alt_train) and os.path.exists(alt_val):
                print(f"找到数据集在: {alt_path}")
                train_dir = alt_train
                val_dir = alt_val
                dataset_exists = True
                found_alternative = True
                break
        
        if not found_alternative:
            print("找不到Imagenette数据集，尝试自动下载...")
            download_success = download_imagenette(data_root)
            
            if not download_success:
                print("自动下载失败")
                print("\n手动下载指南:")
                print("1. 访问: https://github.com/fastai/imagenette")
                print("2. 下载 imagenette2-320.tgz")
                print(f"3. 解压到: {data_root}")
                print("4. 确保目录结构为: {data_root}/train/ 和 {data_root}/val/")
                sys.exit(1)
            
            # 重新检查数据集
            dataset_exists = os.path.exists(train_dir) and os.path.exists(val_dir)
            
            if not dataset_exists:
                print("下载后仍无法找到数据集")
                print(f"期望的训练目录: {train_dir}")
                print(f"期望的验证目录: {val_dir}")
                sys.exit(1)
    else:
        print(f"找到已有数据集: {data_root}")
    
    print("\n3. 加载Imagenette数据集...")
    
    # 使用ImageFolder加载数据集
    try:
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform_train
        )
        
        test_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=transform_test
        )
        
        print(f"Imagenette数据集加载成功!")
        print(f"类别数量: {len(train_dataset.classes)}")
        print(f"类别列表: {train_dataset.classes}")
        
        # 验证数据集
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("数据集为空")
        
        if len(train_dataset.classes) != 10:
            print(f"警告: 期望10个类别，实际找到{len(train_dataset.classes)}个类别")
        
        # 打印类别映射
        print(f"类别到索引的映射: {train_dataset.class_to_idx}")
        
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

def create_models():
    """创建模型 - 为Imagenette创建模型，确保模型在正确的GPU上"""
    global DEVICE
    output_dim = 64  # 每个底部模型的输出维度
    
    # 确认当前设备
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前CUDA设备: {current_device} - {torch.cuda.get_device_name(current_device)}")
        if current_device != args.gpu:
            print(f"警告: 当前激活的设备({current_device})与请求的设备({args.gpu})不匹配")
            print("尝试切换到指定设备...")
            torch.cuda.set_device(args.gpu)
            print(f"已切换到设备: {torch.cuda.current_device()}")
    
    # 创建底部模型
    bottom_models = []
    for i in range(args.party_num):
        if i == args.bkd_adversary:
            # 创建恶意模型
            model = BottomModelResNet(
                output_dim=output_dim,
                party_num=args.party_num,
                is_adversary=True,
                args=args
            )
        else:
            # 创建正常模型
            model = BottomModelResNet(
                output_dim=output_dim,
                party_num=args.party_num
            )
        
        # 立即将模型移动到GPU
        if torch.cuda.is_available():
            model = model.to(DEVICE)
            print(f"模型 {i} 已移动到设备: {next(model.parameters()).device}")
        
        bottom_models.append(model)
    
    # 创建顶部模型
    modelC = ImagenetteTopModel(
        input_dim=output_dim * args.party_num,
        num_classes=args.num_classes
    )
    
    if torch.cuda.is_available():
        modelC = modelC.to(DEVICE)
        print(f"顶部模型已移动到设备: {next(modelC.parameters()).device}")
    
    # 创建并设置BadVFL触发器
    badvfl_trigger = BadVFLTrigger(args)
    if torch.cuda.is_available():
        # 确保触发器的pattern_mask也在正确的设备上
        if hasattr(badvfl_trigger, 'pattern_mask'):
            badvfl_trigger.pattern_mask = badvfl_trigger.pattern_mask.to(DEVICE)
            print(f"触发器模式已移动到: {badvfl_trigger.pattern_mask.device}")
            
    bottom_models[args.bkd_adversary].set_badvfl_trigger(badvfl_trigger)
    
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
    
    # 大幅降低攻击比例，使其保持在5%-10%之间的随机值
    # Imagenette的数据集更复杂，使用更小的攻击比例以避免干扰正常训练
    attack_ratio = random.uniform(0.05, 0.1)
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

def save_checkpoint(modelC, bottom_models, optimizers, optimizerC, epoch, test_acc, true_asr=None, test_inference_acc=None):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    temp = 'ALL' if not args.defense_type=='DPSGD' else 'DPSGD'
    label_knowledge = "True" if args.has_label_knowledge else "False"
    
    if true_asr is None:
        model_name = f"Imagenette_Clean_{temp}_{label_knowledge}_{args.party_num}"
    else:
        model_name = f"Imagenette_BadVFL_WithInference_{temp}_{label_knowledge}_{args.party_num}"
    
    model_file_name = f"{model_name}.pth"
    model_save_path = os.path.join(args.checkpoint_dir, model_file_name)
    
    checkpoint = {
        'model_bottom': {f'bottom_model_{i}': model.state_dict() for i, model in enumerate(bottom_models)},
        'model_top': modelC.state_dict(),
        'epoch': epoch,
        'clean_acc': test_acc,
        'asr': true_asr,
        'inference_acc': test_inference_acc,
        'attack_type': 'BadVFL_WithInference',
        'trigger_magnitude': args.trigger_intensity,
        'trigger_size': args.trigger_size,
        'poison_budget': args.poison_budget,
        'inference_weight': args.inference_weight
    }
    
    torch.save(checkpoint, model_save_path)
    print(f'保存模型到 {model_save_path}')

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
    batch = torch.randn(16, 3, 224, 224, device=DEVICE)
    
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

def test_gpu_working():
    """测试GPU是否正常工作"""
    if not torch.cuda.is_available():
        print("没有可用的GPU!")
        return False
    
    try:
        # 设置当前设备
        torch.cuda.set_device(args.gpu)
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

def train_epoch(modelC, bottom_models, optimizers, optimizerC, train_loader, epoch, args, label_inference_module=None, defense_hooks=None):
    """训练一个轮次，包括BadVFL后门注入和标签推断 - 优化版本，确保正确使用GPU"""
    global DEVICE
    
    # 确保所有模型都处于训练模式
    modelC.train()
    for model in bottom_models:
        model.train()
    
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
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # 确定是否应用后门攻击
    apply_backdoor = epoch >= args.Ebkd
    
    has_inference = hasattr(adversary_model, 'label_inference') and label_inference_module is not None
    if has_inference:
        print(f"标签推断模块状态: {'已初始化' if label_inference_module.initialized else '未初始化'}")
        history_len = len(label_inference_module.history_features_cpu) if hasattr(label_inference_module, 'history_features_cpu') else 0
        print(f"当前收集到的样本数: {history_len}")
    
    # 记录批次数
    batch_count = 0
    # 增加早期阶段的收集批次数量
    warmup_batches = min(100 if epoch < args.Ebkd else 50, len(train_loader) // 10)
    
    # 修改后门损失的权重 - 使用更低的初始权重和更慢的增长速度
    if epoch >= args.Ebkd:
        # 初始权重大幅降低为原始权重的30%
        base_weight = args.backdoor_weight * 0.3
        
        # 使用更缓和的增长曲线
        # 起始阶段（前10个epoch)增长非常缓慢
        if epoch - args.Ebkd < 10:
            epoch_factor = 0.01 * (epoch - args.Ebkd)  # 每个epoch只增加1%
        else:
            # 之后增长速度略微加快，但仍然缓慢
            epoch_factor = 0.1 + 0.005 * (epoch - args.Ebkd - 10)  # 基础增加10%后，每个epoch增加0.5%
        
        # 将最高增幅限制在1.5倍
        backdoor_weight_multiplier = min(1.5, 1.0 + epoch_factor)
        backdoor_weight = base_weight * backdoor_weight_multiplier
        
        # 计算当前权重与干净损失的比例
        weight_ratio = backdoor_weight / 1.0
        
        # 打印调试信息
        initial_weight = args.backdoor_weight * 0.3  # 初始权重
        max_weight = initial_weight * 1.5  # 最大权重
        print(f"当前后门损失权重: {backdoor_weight:.2f} (初始:{initial_weight:.2f}, 最大:{max_weight:.2f}, 比例:{weight_ratio:.2f})")
    else:
        backdoor_weight_multiplier = 0.0
        backdoor_weight = 0.0
    
    if apply_backdoor:
        print(f"当前后门损失权重: {backdoor_weight:.2f}")
    else:
        print(f"当前为正常训练阶段，尚未开始后门攻击 (开始轮次: {args.Ebkd})")
        if has_inference:
            print(f"标签推断模块状态: {'已初始化' if label_inference_module.initialized else '未初始化'}")
            history_len = len(label_inference_module.history_features_cpu) if hasattr(label_inference_module, 'history_features_cpu') else 0
            print(f"当前收集到的样本数: {history_len}")
    
    # 不再使用预取策略，直接使用简单的数据加载
    prefetch_data = None
    prefetch_target = None
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc=f"训练 (Epoch {epoch})")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 增加批次计数
        batch_count += 1
        
        # 定期清理显存，防止碎片化
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 确保数据在GPU上 - 显式检查和传输
        if data.device != DEVICE:
            data = data.to(DEVICE, non_blocking=True)
        if target.device != DEVICE:
            target = target.to(DEVICE, non_blocking=True)
            
        # 确保数据是浮点类型
        if data.dtype != torch.float32:
            data = data.float()
            
        # 记录数据设备位置(测试用)
        if batch_idx == 0:
            print(f"批次数据设备: {data.device}, 标签设备: {target.device}")
            
        # 清空梯度
        for optimizer in optimizers:
            optimizer.zero_grad()
        optimizerC.zero_grad()
        
        total += len(data)
        
        # 如果使用梯度累积，只有在需要更新时才清零梯度
        if batch_count % grad_accumulation_steps == 1:
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
        
        # 默认使用干净损失
        loss = loss_clean
        
        # 前向传播 - 只有在开始后门攻击后才注入后门触发器
        if apply_backdoor:
            # 检查标签推断模块是否已初始化
            has_inference = (hasattr(adversary_model, 'label_inference') and 
                            adversary_model.label_inference is not None and 
                            adversary_model.label_inference.initialized)
            
            if not has_inference:
                print("警告: 标签推断未初始化，跳过后门攻击")
                
            else: # has_inference is True
                # 使用标签推断模块确定要攻击哪些样本
                with torch.no_grad():  # 推断不需要梯度
                    # 使用增强特征用于标签推断
                    try:
                        features = get_enhanced_features(bottom_models, data)
                    except Exception as e:
                        print(f"增强特征提取失败，回退到layer3特征: {str(e)}")
                        features = get_layer3_features(bottom_models, data)
                    
                    inferred_labels, inferred_confidence = adversary_model.label_inference.infer_labels(
                        features, modelC, bottom_models, data
                    )
                    
                    # 如果推断失败，跳过后门攻击
                    if inferred_labels is None:
                        print("警告: 标签推断失败，跳过后门攻击")
                        attack_flags = torch.zeros(len(data), dtype=torch.bool, device=DEVICE) # no attack
                    else:
                        # 计算标签推断的当前批次准确率
                        true_binary_labels = torch.tensor([(1 if t.item() == args.target_class else 0) 
                                                        for t in target], device=DEVICE)
                        correct_inferences = (torch.tensor(inferred_labels, device=DEVICE) == true_binary_labels).float()
                        batch_inference_acc = correct_inferences.mean().item() * 100
                        print(f"批次 {batch_idx} 标签推断准确率: {batch_inference_acc:.2f}%")
                        
                        # 生成attack_flags（基于标签推断结果）
                        attack_flags = torch.zeros(len(data), dtype=torch.bool, device=DEVICE)
                        for i in range(len(data)):
                            if inferred_labels[i] == 0 and inferred_confidence[i] < 0.3:
                                attack_flags[i] = True
                        # 限制攻击样本比例，最多5%（至少1个）
                        max_attack = max(1, int(0.05 * len(data)))
                        attack_indices = torch.where(attack_flags)[0]
                        if len(attack_indices) > max_attack:
                            selected = np.random.choice(attack_indices.cpu().numpy(), max_attack, replace=False)
                            new_attack_flags = torch.zeros_like(attack_flags)
                            new_attack_flags[selected] = True
                            attack_flags = new_attack_flags
                        print(f"批次 {batch_idx}: 总样本={len(data)}, 推断为非目标类={np.sum(np.array(inferred_labels) == 0)}, 攻击样本={attack_flags.sum().item()}")
                
                backdoor_samples += attack_flags.sum().item()
            
                # 准备后门数据
                bkd_data = data.clone()
                bkd_target = target.clone()
                
                # 修改攻击样本的标签为目标类
                bkd_target[attack_flags] = args.target_class
                
                # 注入触发器 - 使用标签推断确定的攻击样本，同时传递推断置信度
                if adversary_model.badvfl_trigger is not None:
                    bkd_data = adversary_model.badvfl_trigger.inject_trigger(
                        bkd_data, 
                        attack_flags, 
                        confidence=inferred_confidence if inferred_labels is not None else None
                    )
                
                bottom_outputs_bkd = []
                for i, model in enumerate(bottom_models):
                    output = model(bkd_data)
                    bottom_outputs_bkd.append(output)
                
                combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                output_bkd = modelC(combined_output_bkd)
                loss_bkd = criterion(output_bkd, bkd_target)
                
                # 组合损失 - 使基于标签推断准确率动态调整后门损失权重
                # 标签推断准确率越高，后门损失权重越大
                inference_weight_factor = min(1.0, batch_inference_acc / 100.0) if inferred_labels is not None else 0.0 # 0.0-1.0
                adjusted_backdoor_weight = backdoor_weight * inference_weight_factor
                
                print(f"批次 {batch_idx}: 标签推断准确率={(batch_inference_acc if inferred_labels is not None else 'N/A'):.2f}%, 调整后的后门权重={adjusted_backdoor_weight:.4f}")
                
                # 使用动态权重平衡干净损失和后门损失
                clean_weight = 1.0  # 固定干净损失的权重
                loss = (clean_weight * loss_clean + adjusted_backdoor_weight * loss_bkd) / (clean_weight + adjusted_backdoor_weight)
                
                # 计算后门准确率 - 只考虑被攻击的样本
                pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                
                # 获取被攻击样本的预测和目标并比较
                attack_success = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags]))
                current_backdoor_correct = attack_success.sum().item()
                backdoor_correct += current_backdoor_correct
                
                # 打印当前批次的攻击成功率
                if attack_flags.sum().item() > 0:
                    current_asr = 100.0 * current_backdoor_correct / attack_flags.sum().item()
                    print(f"批次 {batch_idx} ASR: {current_asr:.2f}% ({current_backdoor_correct}/{attack_flags.sum().item()})")
        
        # backward
        if defense_hooks and defense_hooks.get('pre_backward_hook'):
            defense_hooks['pre_backward_hook']()

        loss.backward()

        if defense_hooks and defense_hooks.get('post_backward_hook'):
            defense_hooks['post_backward_hook'](
                bottom_models=bottom_models,
                modelC=modelC
            )

        torch.nn.utils.clip_grad_norm_(modelC.parameters(), args.grad_clip)
        for model in bottom_models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if defense_hooks and defense_hooks.get('optimizer_step_hook'):
            defense_hooks['optimizer_step_hook'](
                optimizers=[*optimizers, optimizerC],
                bottom_models=bottom_models,
                modelC=modelC
            )
        else:
            for optimizer in optimizers:
                optimizer.step()
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
        if batch_idx % args.log_interval == 0:
            progress = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({progress:.0f}%)]\tLoss: {loss.item() * grad_accumulation_steps:.6f}')
            # 添加GPU内存使用监控
            if torch.cuda.is_available():
                print(get_gpu_memory_usage())
    
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
        if epoch > args.Ebkd + 10 and attack_success_rate > 95:
            # 对于后期训练，如果ASR已经很高，轻微降低后门权重
            backdoor_weight = backdoor_weight * 0.9
            print(f"检测到高ASR，降低后门权重至: {backdoor_weight:.2f}")
    
    # 测试标签推断性能
    inference_accuracy = 0.0
    if has_inference and label_inference_module and label_inference_module.initialized:
        # 创建测试子集
        print("\n===== 开始评估标签推断准确率 =====")
        test_subset_size = min(1000, len(train_loader.dataset))  # 增加样本数以获得更准确的评估
        test_subset_loader = torch.utils.data.Subset(
            train_loader.dataset, 
            indices=range(test_subset_size)
        )
        test_subset_loader = torch.utils.data.DataLoader(
            test_subset_loader, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0  # 禁用多进程以避免CUDA错误
        )
        
        # 用于评估推断性能的统计变量
        correct_predictions = 0
        total_samples = 0
        
        # 详细统计
        class_stats = {
            'target': {'correct': 0, 'total': 0},
            'non_target': {'correct': 0, 'total': 0}
        }
        
        print(f"评估样本总数: {test_subset_size}")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_subset_loader):
                # 定期清理显存，防止碎片化
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 确保数据在正确的设备上
                if data.device != DEVICE:
                    data = data.to(DEVICE)
                if target.device != DEVICE:
                    target = target.to(DEVICE)
                
                # 打印批次进度
                if batch_idx % 5 == 0:
                    print(f"处理批次 {batch_idx}/{len(test_subset_loader)}")
                
                # 打印第一个批次的标签分布
                if batch_idx == 0:
                    target_class_count = (target == args.target_class).sum().item()
                    print(f"标签分布：目标类 ({args.target_class}) 样本数: {target_class_count}, 非目标类样本数: {len(target) - target_class_count}")
                
                # 使用标签推断预测 - 直接使用原始图像数据
                try:
                    features = get_enhanced_features(bottom_models, data)
                except Exception as e:
                    print(f"评估阶段增强特征提取失败，回退到layer3特征: {str(e)}")
                    features = get_layer3_features(bottom_models, data)
                    
                inferred_labels, inferred_confidence = label_inference_module.infer_labels(features)
                
                if inferred_labels is not None:
                    # 计算推断准确率 - 使用二分类计算方式
                    for j, (pred, true) in enumerate(zip(inferred_labels, target.cpu().numpy())):
                        # 确定真实的二分类标签
                        is_target_class = (true == args.target_class)
                        true_binary = 1 if is_target_class else 0
                        
                        # 更新类别统计
                        if is_target_class:
                            class_stats['target']['total'] += 1
                            if pred == 1:  # 正确预测为目标类
                                class_stats['target']['correct'] += 1
                        else:
                            class_stats['non_target']['total'] += 1
                            if pred == 0:  # 正确预测为非目标类
                                class_stats['non_target']['correct'] += 1
                        
                        # 判断预测是否正确
                        is_correct = (pred == true_binary)
                        if is_correct:
                            correct_predictions += 1
                        
                        # 详细记录前20个样本的推断结果
                        if total_samples < 20:
                            print(f"样本 {total_samples}: 真实类别={true}, 二分类标签={true_binary}, 预测={pred}, 正确={is_correct}")
                        
                        total_samples += 1
                else:
                    print("警告: infer_labels返回了None，无法计算标签推断准确率")
                    # 如果推断失败但已经处理了一些样本，尝试继续
                    if total_samples > 0:
                        continue
                    else:
                        # 如果一个样本都没处理，设置一个默认值
                        print("设置默认标签推断准确率为0%")
                        return avg_loss, accuracy, attack_success_rate, 0.0
        
            # 计算推断准确率
            if total_samples > 0:
                inference_accuracy = 100.0 * correct_predictions / total_samples
                
                # 计算每个类别的准确率
                target_acc = 0
                if class_stats['target']['total'] > 0:
                    target_acc = 100.0 * class_stats['target']['correct'] / class_stats['target']['total']
                    
                non_target_acc = 0
                if class_stats['non_target']['total'] > 0:
                    non_target_acc = 100.0 * class_stats['non_target']['correct'] / class_stats['non_target']['total']
                
                # 打印详细结果
                print("\n===== 标签推断准确率详情 =====")
                print(f"总体准确率: {correct_predictions}/{total_samples} = {inference_accuracy:.2f}%")
                print(f"目标类准确率: {class_stats['target']['correct']}/{class_stats['target']['total']} = {target_acc:.2f}%")
                print(f"非目标类准确率: {class_stats['non_target']['correct']}/{class_stats['non_target']['total']} = {non_target_acc:.2f}%")
                print("================================")
            else:
                inference_accuracy = 0.0
                print("警告: 没有样本用于计算标签推断准确率")
    else:
        if has_inference:
            if not label_inference_module:
                print("警告: 标签推断模块不存在")
            elif not label_inference_module.initialized:
                print("警告: 标签推断模块未初始化")
        else:
            print("警告: 没有标签推断功能，设置标签推断准确率为0")
        inference_accuracy = 0.0
    
    # 清理缓存以释放GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, accuracy, attack_success_rate, inference_accuracy

def test(modelC, bottom_models, test_loader, is_backdoor=False, epoch=0, args=None, train_loader=None, defense_hooks=None):
    """测试模型性能，包括干净准确率和后门攻击成功率 - 优化版本，确保正确使用GPU"""
    global DEVICE
    
    print("=" * 80)
    print("【TEST函数被调用】 - 这是一个强制调试输出")
    print(f"epoch={epoch}, is_backdoor={is_backdoor}, args.Ebkd={args.Ebkd}")
    print(f"train_loader是否为None: {train_loader is None}")
    print("=" * 80)
    
    # 确保模型处于评估模式
    modelC.eval()
    for model in bottom_models:
        model.eval()
    
    # 新增：测试阶段自动修复标签推断未初始化
    adversary_model = bottom_models[args.bkd_adversary]

    # 添加详细的调试信息
    print(f"\n[测试阶段调试] 恶意模型是否有label_inference属性: {hasattr(adversary_model, 'label_inference')}")
    if hasattr(adversary_model, 'label_inference'):
        print(f"[测试阶段调试] label_inference是否为None: {adversary_model.label_inference is None}")
        if adversary_model.label_inference is not None:
            print(f"[测试阶段调试] label_inference是否已初始化: {adversary_model.label_inference.initialized}")
        else:
            print("[测试阶段调试] label_inference为None，需要重新创建")
    else:
        print("[测试阶段调试] 恶意模型没有label_inference属性")

    # 修复自动修复逻辑：无论label_inference是None还是未初始化都触发修复
    needs_repair = False
    if not hasattr(adversary_model, 'label_inference'):
        print("[测试阶段] 恶意模型缺少label_inference属性")
        needs_repair = True
    elif adversary_model.label_inference is None:
        print("[测试阶段] label_inference为None，需要重新创建")
        needs_repair = True
    elif not adversary_model.label_inference.initialized:
        print("[测试阶段] label_inference未初始化")
        needs_repair = True

    if needs_repair:
        if train_loader is not None:
            print("[自动修复] 测试阶段标签推断未初始化，自动重新收集标签推断数据...")
            collect_inference_data(modelC, bottom_models, train_loader, args)
            # 重新检查修复结果
            if hasattr(adversary_model, 'label_inference') and adversary_model.label_inference is not None:
                print(f"[自动修复结果] label_inference是否已初始化: {adversary_model.label_inference.initialized}")
            else:
                print("[自动修复结果] 修复失败，label_inference仍为None")
        else:
            print("警告: 测试阶段标签推断未初始化，且未传入train_loader，将导致ASR=0。")
    
    # 启用cudnn基准测试提高性能
    torch.backends.cudnn.benchmark = True
    
    # 定义apply_backdoor变量，与train_epoch函数中保持一致
    apply_backdoor = epoch >= args.Ebkd
    
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
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(test_loader, desc="测试" + (" (带后门)" if is_backdoor else ""))
    
    # 极大减小测试中的攻击样本比例，只攻击3%的非目标类样本
    MAX_TEST_ATTACK_RATIO = 0.03
    
    # 设置最大总攻击样本数，避免测试集中使用太多样本
    MAX_TOTAL_ATTACK_SAMPLES = 500
    current_total_attack_samples = 0
    
    # 随机选择批次，避免偏向特定位置的批次
    batch_indices = list(range(len(test_loader)))
    random.shuffle(batch_indices)
    selected_batch_indices = set(batch_indices[:min(50, len(batch_indices))])
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            # 定期清理显存，防止碎片化
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 确保数据在正确的设备上
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = data.size(0)
            total += batch_size
            
            # 记录非目标类样本数量 - 用于调试
            current_non_target = (target != args.target_class).sum().item()
            non_target_total += current_non_target
            if current_non_target > 0:
                non_target_batches += 1
            
            # 准备干净数据的预测
            bottom_outputs_clean = []
            for model in bottom_models:
                output = model(data)
                bottom_outputs_clean.append(output)
            
            combined_output_clean = torch.cat(bottom_outputs_clean, dim=1)
            output_clean = modelC(combined_output_clean)
            
            # 计算损失和准确率
            test_loss += criterion(output_clean, target).item()
            pred_clean = output_clean.argmax(dim=1, keepdim=True)
            correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()
            
            # 如果需要测试后门攻击
            if is_backdoor and apply_backdoor:
                # 检查标签推断模块是否已初始化
                has_inference = (hasattr(adversary_model, 'label_inference') and 
                                adversary_model.label_inference is not None and 
                                adversary_model.label_inference.initialized)
                
                # 如果标签推断未初始化，则无法测试ASR
                if not has_inference:
                    print("警告: 标签推断未初始化，无法评估ASR")
                    continue
                
                # 使用标签推断模块确定要攻击哪些样本 - 与训练阶段保持一致
                with torch.no_grad():  # 推断不需要梯度
                    # 使用与训练阶段相同的增强特征提取方法
                    try:
                        features = get_enhanced_features(bottom_models, data)
                    except Exception as e:
                        print(f"测试阶段增强特征提取失败，回退到layer3特征: {str(e)}")
                        features = get_layer3_features(bottom_models, data)
                    
                    # 使用标签推断模块预测样本类别
                    inferred_labels, inferred_confidence = adversary_model.label_inference.infer_labels(
                        features, modelC, bottom_models, data
                    )
                    
                    # 如果推断失败，跳过当前批次
                    if inferred_labels is None:
                        print("标签推断失败，跳过当前批次")
                        continue
                    
                    # 计算标签推断准确率
                    true_binary_labels = torch.tensor([(1 if t.item() == args.target_class else 0) 
                                                     for t in target], device=DEVICE)
                    correct_inferences = (torch.tensor(inferred_labels, device=DEVICE) == true_binary_labels).float()
                    batch_inference_acc = correct_inferences.mean().item() * 100
                    print(f"测试批次 {batch_idx} 标签推断准确率: {batch_inference_acc:.2f}%")
                    
                    # 记录标签推断准确率的累积平均值
                    if 'total_inference_acc' not in locals():
                        total_inference_acc = 0
                        total_inference_batches = 0
                    total_inference_acc += batch_inference_acc
                    total_inference_batches += 1
                    
                    # 创建基于推断的攻击标志 - 只攻击推断为非目标类(0)的样本
                    attack_flags = torch.zeros(len(data), dtype=torch.bool, device=DEVICE)
                    for i in range(len(data)):
                        # 只有当推断为非目标类且推断置信度高于阈值时才攻击
                        if inferred_labels[i] == 0 and inferred_confidence[i] < 0.3:  # 非目标类且高置信度
                            attack_flags[i] = True
                    
                    # 计算当前批次的攻击样本数
                    current_attack_count = attack_flags.sum().item()
                    
                    # 如果没有识别出任何非目标类样本，跳过当前批次
                    if current_attack_count == 0:
                        print(f"测试批次 {batch_idx}: 未找到合适的攻击样本，跳过")
                        continue
                    
                    # 更新总攻击样本计数
                    backdoor_samples += current_attack_count
                    
                    # 打印当前批次的攻击情况
                    print(f"测试批次 {batch_idx}: 总样本={len(data)}, 推断为非目标类={np.sum(np.array(inferred_labels) == 0)}, 攻击样本={current_attack_count}")
                
                # 准备后门数据
                bkd_data = data.clone()
                bkd_target = target.clone()
                
                # 修改攻击样本的标签为目标类
                bkd_target[attack_flags] = args.target_class
                
                # 注入触发器
                if hasattr(adversary_model, 'badvfl_trigger') and adversary_model.badvfl_trigger is not None:
                    # 注入触发器 - 同样仅使用标签推断结果确定攻击样本，同时传递置信度信息
                    bkd_data = adversary_model.badvfl_trigger.inject_trigger(
                        bkd_data, 
                        attack_flags, 
                        confidence=inferred_confidence  # 传递置信度信息
                    )
                
                # 前向传播 - 对包含注入触发器的样本
                bottom_outputs_bkd = []
                for model in bottom_models:
                    output = model(bkd_data)
                    bottom_outputs_bkd.append(output)
                
                combined_output_bkd = torch.cat(bottom_outputs_bkd, dim=1)
                output_bkd = modelC(combined_output_bkd)
                
                # 计算攻击成功的样本数 - 只考虑被修改标签的样本
                pred_bkd = output_bkd.argmax(dim=1, keepdim=True)
                
                # 验证攻击效果
                attack_success = pred_bkd[attack_flags].eq(bkd_target[attack_flags].view_as(pred_bkd[attack_flags]))
                current_backdoor_correct = attack_success.sum().item()
                backdoor_correct += current_backdoor_correct
                
                # 打印当前批次的攻击成功率
                current_asr = 100.0 * current_backdoor_correct / current_attack_count if current_attack_count > 0 else 0
                print(f"批次 {batch_idx} ASR: {current_asr:.2f}% ({current_backdoor_correct}/{current_attack_count})")
    
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
    """收集标签推断数据，使用增强特征提取以提高推断准确率 - 改进版本"""
    print("\n===== 开始收集标签推断数据 (增强版本v2) =====")
    modelC.eval()  # 使用评估模式获取更稳定的特征
    for model in bottom_models:
        model.eval()
    adversary_model = bottom_models[args.bkd_adversary]
    label_inference_module = adversary_model.label_inference
    if not label_inference_module:
        print("错误: 恶意模型没有标签推断模块")
        return False
    
    # 更加平衡的数据收集策略
    min_target = 400        # 增加目标类样本数
    min_non_target = 800    # 增加非目标类样本数  
    max_total = 8000        # 增加总样本数
    max_batches = 300       # 增加批次数
    
    current_samples = len(label_inference_module.history_features_cpu) if hasattr(label_inference_module, 'history_features_cpu') else 0
    if current_samples > 0:
        print(f"清空已有的 {current_samples} 个样本，重新收集...")
        # 清空现有数据，重新收集更好的数据
        label_inference_module.history_features_cpu = []
        label_inference_module.history_predictions_cpu = []
    
    if hasattr(label_inference_module, 'min_samples'):
        old_min = label_inference_module.min_samples
        label_inference_module.min_samples = 50  # 提高最小样本需求
        print(f"设置标签推断模块最小样本需求: {old_min} -> 50")
    
    class_counts = {'target': 0, 'non_target': 0}
    total_collected = 0
    batches_processed = 0
    sample_features = []
    sample_labels = []  # 1=目标类，0=非目标类
    
    print(f"目标: 收集至少 {min_target} 个目标类样本和 {min_non_target} 个非目标类样本")
    progress_bar = tqdm(total=max_batches, desc="收集高质量标签推断数据")
    
    # 使用多种特征提取方法
    all_batch_data = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if (class_counts['target'] >= min_target and 
                class_counts['non_target'] >= min_non_target) or \
                total_collected >= max_total or \
                batches_processed >= max_batches:
                break
            
            batches_processed += 1
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 收集这个批次的数据用于后续处理
            all_batch_data.append((data.clone(), target.clone()))
            
            # 使用多种特征提取方法的组合
            feature_sets = []
            
            # 方法1: 增强特征
            try:
                enhanced_features = get_enhanced_features(bottom_models, data)
                feature_sets.append(enhanced_features)
                print(f"批次 {batch_idx}: 增强特征形状={enhanced_features.shape}")
            except Exception as e:
                print(f"增强特征提取失败: {str(e)}")
            
            # 方法2: layer3特征作为备用
            try:
                layer3_features = get_layer3_features(bottom_models, data)
                if len(feature_sets) == 0:  # 只有在增强特征失败时才使用
                    feature_sets.append(layer3_features)
                    print(f"批次 {batch_idx}: 使用layer3特征，形状={layer3_features.shape}")
            except Exception as e:
                print(f"layer3特征提取失败: {str(e)}")
            
            if len(feature_sets) == 0:
                print(f"批次 {batch_idx}: 所有特征提取方法都失败，跳过")
                continue
            
            # 使用最好的特征集
            best_features = feature_sets[0]
            
            target_class_count = (target == args.target_class).sum().item()
            non_target_count = len(target) - target_class_count
            print(f"批次 {batch_idx}: 总样本={len(target)}, 目标类={target_class_count}, 非目标类={non_target_count}")
            
            # 平衡采样，确保类别分布合理
            for i in range(len(target)):
                sample_data = best_features[i].cpu().numpy()
                true_label = target[i].item()
                binary_label = 1 if true_label == args.target_class else 0
                class_key = 'target' if binary_label == 1 else 'non_target'
                
                if np.isnan(sample_data).any() or np.isinf(sample_data).any():
                    print(f"警告: 样本 {i} 包含NaN或Inf值，跳过")
                    continue
                
                # 确保数据质量和平衡性
                current_ratio = class_counts['target'] / max(1, class_counts['target'] + class_counts['non_target'])
                target_ratio = min_target / (min_target + min_non_target)
                
                # 根据当前分布调整采样策略
                should_collect = False
                if class_key == 'target':
                    if class_counts['target'] < min_target:
                        should_collect = True
                    elif current_ratio < target_ratio and random.random() < 0.3:
                        should_collect = True
                else:  # non_target
                    if class_counts['non_target'] < min_non_target:
                        should_collect = True
                    elif current_ratio > target_ratio and random.random() < 0.7:
                        should_collect = True
                
                if should_collect:
                    # 添加轻微的数据增强以提高泛化性
                    augmented_data = sample_data.copy()
                    if random.random() < 0.3:  # 30%概率添加轻微噪声
                        noise_scale = 0.01 * np.std(sample_data)
                        noise = np.random.normal(0, noise_scale, sample_data.shape)
                        augmented_data += noise
                    
                    sample_features.append(augmented_data)
                    sample_labels.append(binary_label)
                    class_counts[class_key] += 1
                    total_collected += 1
            
            progress_bar.update(1)
            progress_bar.set_postfix(
                total=total_collected, 
                target=class_counts['target'], 
                non_target=class_counts['non_target'],
                ratio=f"{class_counts['target']/max(1,total_collected):.2f}"
            )
    
    progress_bar.close()
    print(f"高质量数据收集完成: 总计 {total_collected} 个样本")
    print(f"目标类: {class_counts['target']}/{min_target}")
    print(f"非目标类: {class_counts['non_target']}/{min_non_target}")
    
    if total_collected > 0:
        features_array = np.array(sample_features)
        labels_array = np.array(sample_labels)
        print(f"最终特征形状: {features_array.shape}")
        print(f"标签形状: {labels_array.shape}")
        print(f"标签分布: 目标类(1)={np.sum(labels_array == 1)}, 非目标类(0)={np.sum(labels_array == 0)}")
        print(f"类别平衡性: {np.sum(labels_array == 1) / len(labels_array):.3f}")
        
        # 数据质量检查
        print(f"特征统计: 均值={np.mean(features_array):.4f}, 标准差={np.std(features_array):.4f}")
        print(f"特征范围: [{np.min(features_array):.4f}, {np.max(features_array):.4f}]")
        
        # 重新初始化标签推断模块，传入正确的特征维度
        adversary_model.label_inference = BadVFLLabelInference(
            feature_dim=features_array.shape[1],
            num_classes=args.num_classes,
            args=args
        )
        label_inference_module = adversary_model.label_inference
        
        # 清空历史记录并添加新数据
        label_inference_module.history_features_cpu = features_array.tolist()
        label_inference_module.history_predictions_cpu = labels_array.tolist()
        label_inference_module.history_features_gpu = torch.FloatTensor(features_array).to(DEVICE)
        label_inference_module.history_predictions_gpu = torch.FloatTensor(labels_array).to(DEVICE)
        
        print(f"历史记录已更新，CPU版本: {len(label_inference_module.history_features_cpu)} 个样本")
        print(f"历史记录已更新，GPU版本: {label_inference_module.history_features_gpu.shape[0]} 个样本")
        
        max_attempts = 5
        for attempt in range(max_attempts):
            print(f"尝试初始化高质量标签推断分类器... (尝试 {attempt+1}/{max_attempts})")
            if label_inference_module.initialize_classifier():
                print("高质量标签推断分类器初始化成功!")
                if hasattr(adversary_model, 'badvfl_trigger') and adversary_model.badvfl_trigger:
                    adversary_model.badvfl_trigger.update_inference_stats()
                return True
            else:
                print(f"初始化尝试 {attempt+1} 失败")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # 等待1秒后重试
        
        print("所有初始化尝试均失败，但数据已收集，将在训练过程中重试")
        return False
    else:
        print("未收集到样本，标签推断初始化失败")
        return False

def get_layer3_features(bottom_models, data):
    """原始layer3特征提取（保留兼容性）"""
    features_list = []
    for model in bottom_models:
        x = data
        x = model.model.conv1(x)
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        f = model.model.layer3(x)
        f = torch.flatten(f, 1)
        features_list.append(f)
    combined = torch.cat(features_list, dim=1)
    return combined

def get_enhanced_features(bottom_models, data):
    """增强特征提取：使用更深层的特征组合以提高标签推断准确率"""
    features_list = []
    for model in bottom_models:
        x = data
        x = model.model.conv1(x)
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        
        # 提取layer1特征
        x = model.model.layer1(x)
        layer1_feat = F.adaptive_avg_pool2d(x, (4, 4))
        layer1_feat = torch.flatten(layer1_feat, 1)
        
        # 提取layer2特征  
        x = model.model.layer2(x)
        layer2_feat = F.adaptive_avg_pool2d(x, (4, 4))
        layer2_feat = torch.flatten(layer2_feat, 1)
        
        # 提取layer3特征
        x = model.model.layer3(x)
        layer3_feat = F.adaptive_avg_pool2d(x, (4, 4))
        layer3_feat = torch.flatten(layer3_feat, 1)
        
        # 提取layer4特征（最深层）
        x = model.model.layer4(x)
        layer4_feat = F.adaptive_avg_pool2d(x, (2, 2))
        layer4_feat = torch.flatten(layer4_feat, 1)
        
        # 全局平均池化特征
        global_feat = F.adaptive_avg_pool2d(x, (1, 1))
        global_feat = torch.flatten(global_feat, 1)
        
        # 组合多层特征，添加全局特征以提高语义信息
        combined_feat = torch.cat([layer1_feat, layer2_feat, layer3_feat, layer4_feat, global_feat], dim=1)
        features_list.append(combined_feat)
    
    final_combined = torch.cat(features_list, dim=1)
    return final_combined

def main():
    """主函数 - 重构版本，确保正确使用GPU"""
    global DEVICE
    
    print("\n======== 开始初始化 ========")
    # 检查CUDA可用性 
    cuda_available = check_cuda()
    
    if not cuda_available:
        print("错误: 无可用GPU，程序将退出")
        return
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 1. 强制设置设备
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device(f"cuda:{args.gpu}")
    print(f"设置全局设备变量: DEVICE = {DEVICE}")
    
    # 2. 测试GPU是否正常工作
    print("\n测试GPU功能...")
    gpu_working = test_gpu_working()
    if not gpu_working:
        print("警告: GPU测试不通过，可能导致性能问题")
        user_input = input("是否继续? (y/n): ")
        if user_input.lower() != 'y':
            print("用户选择终止程序")
            return
    
    # 3. 批处理大小和学习率设置 - Imagenette使用较小的批次
    if args.batch_size > 128:
        print(f"将批处理大小从 {args.batch_size} 减少到 64 以避免GPU内存不足")
        args.batch_size = 64
    
    # 4. 打印训练参数
    print("\n" + "="*50)
    print("训练参数:")
    print(f"GPU设备: {DEVICE} ({torch.cuda.get_device_name(args.gpu)})")
    print(f"批处理大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"轮次: {args.epochs}")
    print(f"后门注入开始轮次: {args.Ebkd}")
    print(f"后门损失权重: {args.backdoor_weight}")
    print("="*50 + "\n")
    
    # 5. 确保参数一致性
    if hasattr(args, 'trigger_size') and hasattr(args, 'pattern_size'):
        if args.trigger_size != args.pattern_size:
            print(f"注意: 统一触发器大小 {args.trigger_size} -> {args.pattern_size}")
            args.pattern_size = args.trigger_size
    
    # 6. 预热GPU
    print("\n预热GPU...")
    warmup_gpu()
    
    # 7. 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 8. 加载数据集
    print("\n加载数据集...")
    train_loader, test_loader = load_dataset("Imagenette", args.data_dir, args.batch_size)
    
    # 8.1 创建用于防御的良性数据加载器
    print("\n8.1 创建用于防御的良性数据加载器...")
    benign_indices = [i for i, target in enumerate(train_loader.dataset.targets) if target != args.target_class]
    benign_subset = Subset(train_loader.dataset, benign_indices)
    benign_loader = DataLoader(benign_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"创建了包含 {len(benign_subset)} 个样本的良性数据加载器。")

    # 9. 创建模型
    print("\n创建模型...")
    bottom_models, modelC = create_models()
    
    # 简化后的验证
    if torch.cuda.is_available():
        for i, model in enumerate(bottom_models):
            device_str = next(model.parameters()).device
            print(f"底部模型 {i}: {device_str}")
        
        device_str = next(modelC.parameters()).device
        print(f"顶部模型: {device_str}")
    
    # 10. 创建优化器
    print("\n创建优化器...")
    # 对于Imagenette使用更小的学习率和更高的动量
    optimizers = [optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model in bottom_models]
    optimizerC = optim.SGD(modelC.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # === 初始化AMP GradScaler ===
    scaler = torch.cuda.amp.GradScaler()

    # 11. 创建防御钩子
    print("\n创建防御钩子...")
    defense_hooks = build_defense(args, benign_loader=benign_loader)
    
    # 12. 收集标签推断数据
    print("\n收集标签推断数据...")
    adversary_model = bottom_models[args.bkd_adversary]
    
    # 确保触发器已设置
    if not hasattr(adversary_model, 'badvfl_trigger') or adversary_model.badvfl_trigger is None:
        print("设置BadVFL触发器...")
        badvfl_trigger = BadVFLTrigger(args)
        adversary_model.set_badvfl_trigger(badvfl_trigger)
    
    # 收集标签推断数据
    collect_inference_data(modelC, bottom_models, train_loader, args)
    
    # 13. 开始训练
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
    print(f"\n{'='*20} 预训练阶段 (1-{args.Ebkd-1}轮) {'='*20}")
    print(f"此阶段专注于提高标签推断准确率，暂不进行后门攻击")
    
    # 启用早停机制
    print(f"Early Stopping: 启用 (patience={args.patience}, 评估指标=0.5*CleanAcc+0.5*ASR)")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n开始 Epoch {epoch}/{args.epochs}")
        
        # 记录GPU内存使用
        if torch.cuda.is_available():
            print(get_gpu_memory_usage())
            
        # 新增：每个epoch前自动检查并收集标签推断数据
        adversary_model = bottom_models[args.bkd_adversary]
        if hasattr(adversary_model, 'label_inference') and (not adversary_model.label_inference.initialized):
            print("[自动修复] 标签推断未初始化，自动重新收集标签推断数据...")
            collect_inference_data(modelC, bottom_models, train_loader, args)
            
        # 更新标签推断状态 - 每个epoch尝试更新
        if epoch >= 2 and hasattr(adversary_model, 'label_inference') and adversary_model.label_inference is not None:
            print("\n检查并更新标签推断状态...")
            if adversary_model.label_inference.initialized:
                print("标签推断已初始化，无需更新")
            else:
                # 强制更新状态
                adversary_model.label_inference.update_class_stats(modelC, bottom_models, force=(epoch % 5 == 0))
                print(f"标签推断状态: {'已初始化' if adversary_model.label_inference.initialized else '未初始化'}")
                print(f"已收集样本数: {adversary_model.label_inference.get_total_samples()}")
        
        # 训练
        train_loss, train_acc, train_asr, train_inference_acc = train_epoch(
            modelC, bottom_models, optimizers, optimizerC, train_loader, epoch, args, adversary_model.label_inference, defense_hooks
        )
        
        # 测试 - 使用返回的字典格式结果
        test_results = test(
            modelC, bottom_models, test_loader, is_backdoor=(epoch >= args.Ebkd), epoch=epoch, args=args, train_loader=train_loader, defense_hooks=defense_hooks
        )
        
        # 从结果字典中提取值
        test_loss = test_results['loss']
        test_acc = test_results['clean_acc']
        true_asr = test_results['asr']
        
        print(f"\nEpoch {epoch} Results:")
        print(f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, ASR {train_asr:.2f}%, Inference Acc {train_inference_acc:.2f}%")
        print(f"Test: Loss {test_loss:.4f}, Acc {test_acc:.2f}%")
        if epoch >= args.Ebkd:
            print(f"Backdoor: ASR {true_asr:.2f}%")
        else:
            print(f"Backdoor attack not active yet (starts at epoch {args.Ebkd})")

        # Check if this is the best model
        # Only consider ASR in combined score if backdoor attack has started
        if epoch >= args.Ebkd:
            combined_score = 0.5 * test_acc + 0.5 * true_asr
            best_combined_score = 0.5 * best_metrics['test_acc'] + 0.5 * best_metrics['asr']
            
            is_best = combined_score > best_combined_score
            
            if is_best:
                best_metrics = {
                    'test_acc': test_acc,
                    'inference_acc': train_inference_acc,
                    'asr': true_asr,
                    'epoch': epoch,
                    'combined_score': combined_score
                }
                no_improvement_count = 0
                print(f"New best combined score: {combined_score:.2f} (0.5*{test_acc:.2f} + 0.5*{true_asr:.2f})")
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} epochs")
                
            # Early stopping check
            if no_improvement_count >= args.patience:
                print(f"\nEarly stopping triggered! Best model at Epoch {best_metrics['epoch']}")
                break
        else:
            # Before backdoor starts, only consider clean accuracy
            if test_acc > best_metrics['test_acc'] or not first_epoch_processed:
                best_metrics = {
                    'test_acc': test_acc,
                    'inference_acc': train_inference_acc,
                    'asr': true_asr,
                    'epoch': epoch,
                    'combined_score': test_acc  # Use only clean acc before backdoor starts
                }
                no_improvement_count = 0
                if first_epoch_processed:
                    print(f"New best clean accuracy: {test_acc:.2f}%")
            else:
                no_improvement_count += 1
                
            first_epoch_processed = True
        
        # Update individual best metrics regardless of combined score
        if test_acc > best_accuracy:
            best_accuracy = test_acc
        if train_inference_acc > best_inference_acc:
            best_inference_acc = train_inference_acc
        if true_asr > best_asr:
            best_asr = true_asr
        
        # Save checkpoint for best epochs
        save_checkpoint_this_epoch = False
        
        if epoch >= args.Ebkd:
            if combined_score > best_combined_score:
                save_checkpoint_this_epoch = True
        else:
            if test_acc > best_metrics['test_acc'] or not first_epoch_processed:
                save_checkpoint_this_epoch = True
        
        if save_checkpoint_this_epoch or epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(
                modelC=modelC,
                bottom_models=bottom_models,
                optimizers=optimizers,
                optimizerC=optimizerC,
                epoch=epoch,
                test_acc=test_acc,
                true_asr=true_asr if epoch >= args.Ebkd else None,
                test_inference_acc=train_inference_acc
            )
            print(f"Checkpoint saved for epoch {epoch}")
        
        # 每个epoch结束时清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU内存已清理")
            print(get_gpu_memory_usage())

    print("\n" + "="*60)
    print("============================================================")
    print(f"Training completed! Best model (Epoch {best_metrics['epoch']}):")
    print(f"Clean Accuracy: {best_metrics['test_acc']:.2f}%")
    print(f"Attack Success Rate: {best_metrics['asr']:.2f}%")
    print(f"Inference Accuracy: {best_metrics['inference_acc']:.2f}%")
    print("============================================================")
    
    return best_metrics['test_acc'], best_metrics['epoch']

# 主函数调用
if __name__ == '__main__':
    try:
        # 在运行主函数前，先运行一个GPU测试
        print("\n========== GPU测试 ==========")
        if torch.cuda.is_available():
            print(f"CUDA可用。设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            
            # 选择第一个可见的设备
            device_id = 0 if args.gpu is None else args.gpu
            torch.cuda.set_device(device_id)
            print(f"使用GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            
            # 运行一个简单的矩阵乘法测试
            print("运行GPU速度测试...")
            try:
                # 创建大矩阵
                a = torch.randn(2000, 2000, device=f"cuda:{device_id}")
                b = torch.randn(2000, 2000, device=f"cuda:{device_id}")
                
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
    except Exception as e:
        import traceback
        print(f"程序运行出错: {str(e)}")
        traceback.print_exc()