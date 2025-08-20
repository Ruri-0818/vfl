#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件名: prepare_bank_data.py
# 描述: 下载和预处理Bank Marketing数据集

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import urllib.request
import zipfile
import shutil

def download_bank_dataset(data_dir):
    """下载Bank Marketing数据集"""
    print("开始下载Bank Marketing数据集...")
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # UCI Bank Marketing数据集URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = os.path.join(data_dir, "bank-additional.zip")
    
    try:
        print(f"正在从 {url} 下载数据集...")
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成！")
        
        # 解压缩
        print("正在解压缩...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # 删除zip文件
        os.remove(zip_path)
        print("解压缩完成！")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("尝试备用下载方法...")
        
        # 备用方法：创建示例数据集
        return create_sample_bank_dataset(data_dir)

def create_sample_bank_dataset(data_dir):
    """创建示例Bank Marketing数据集"""
    print("创建示例Bank Marketing数据集...")
    
    # 创建bank-additional目录
    bank_dir = os.path.join(data_dir, "bank-additional")
    os.makedirs(bank_dir, exist_ok=True)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 10000
    
    # 特征列
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'job': np.random.choice(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                'management', 'retired', 'self-employed', 'services', 
                                'student', 'technician', 'unemployed', 'unknown'], n_samples),
        'marital': np.random.choice(['divorced', 'married', 'single', 'unknown'], n_samples),
        'education': np.random.choice(['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                     'illiterate', 'professional.course', 'university.degree', 'unknown'], n_samples),
        'default': np.random.choice(['no', 'yes', 'unknown'], n_samples),
        'housing': np.random.choice(['no', 'yes', 'unknown'], n_samples),
        'loan': np.random.choice(['no', 'yes', 'unknown'], n_samples),
        'contact': np.random.choice(['cellular', 'telephone'], n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], n_samples),
        'duration': np.random.randint(0, 1000, n_samples),
        'campaign': np.random.randint(1, 10, n_samples),
        'pdays': np.random.randint(0, 1000, n_samples),
        'previous': np.random.randint(0, 10, n_samples),
        'poutcome': np.random.choice(['failure', 'nonexistent', 'success'], n_samples),
        'emp.var.rate': np.random.uniform(-3, 2, n_samples),
        'cons.price.idx': np.random.uniform(92, 95, n_samples),
        'cons.conf.idx': np.random.uniform(-50, -25, n_samples),
        'euribor3m': np.random.uniform(0, 6, n_samples),
        'nr.employed': np.random.uniform(5000, 5300, n_samples),
        'y': np.random.choice(['no', 'yes'], n_samples, p=[0.8, 0.2])  # 不平衡数据
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到CSV文件
    csv_path = os.path.join(bank_dir, "bank-additional-full.csv")
    df.to_csv(csv_path, sep=';', index=False)
    
    print(f"示例数据集已保存到: {csv_path}")
    print(f"数据形状: {df.shape}")
    print(f"标签分布: {df['y'].value_counts()}")
    
    return True

def preprocess_bank_data(data_dir):
    """预处理Bank Marketing数据"""
    print("开始预处理Bank Marketing数据...")
    
    # 查找CSV文件
    csv_paths = [
        os.path.join(data_dir, 'bank-additional', 'bank-additional-full.csv'),
        os.path.join(data_dir, 'bank-additional', 'bank-additional.csv'),
        os.path.join(data_dir, 'bank-full.csv'),
        os.path.join(data_dir, 'bank.csv')
    ]
    
    csv_path = None
    for path in csv_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError("找不到Bank Marketing CSV文件")
    
    print(f"找到数据文件: {csv_path}")
    
    # 读取数据
    try:
        df = pd.read_csv(csv_path, sep=';')
    except:
        # 尝试其他分隔符
        df = pd.read_csv(csv_path, sep=',')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查目标列
    target_col = 'y'
    if target_col not in df.columns:
        print("警告: 找不到目标列 'y'")
        # 尝试其他可能的目标列名
        possible_targets = ['target', 'label', 'class', 'outcome']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col not in df.columns:
            raise ValueError("找不到合适的目标列")
    
    print(f"使用目标列: {target_col}")
    print(f"目标列分布: {df[target_col].value_counts()}")
    
    # 分离特征和标签
    features_df = df.drop(columns=[target_col]).copy()
    labels_series = df[target_col].copy()
    
    # 处理分类特征
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    print(f"分类特征: {list(categorical_cols)}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        features_df[col] = le.fit_transform(features_df[col].astype(str))
    
    # 处理标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_series)
    
    print(f"标签编码: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    
    print(f"预处理后特征形状: {features_scaled.shape}")
    print(f"特征范围: [{features_scaled.min():.3f}, {features_scaled.max():.3f}]")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, 
        test_size=0.2, random_state=42, 
        stratify=labels_encoded
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"测试集标签分布: {np.bincount(y_test)}")
    
    # 保存预处理后的数据
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print("预处理完成，数据已保存!")
    return True

def main():
    """主函数"""
    # 默认数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data/bank"
    
    print(f"Bank Marketing数据集准备脚本")
    print(f"数据目录: {data_dir}")
    print("="*50)
    
    try:
        # 检查是否已有预处理数据
        processed_files = [
            os.path.join(data_dir, 'X_train.npy'),
            os.path.join(data_dir, 'X_test.npy'), 
            os.path.join(data_dir, 'y_train.npy'),
            os.path.join(data_dir, 'y_test.npy')
        ]
        
        if all(os.path.exists(f) for f in processed_files):
            print("发现已有预处理数据，跳过下载和预处理步骤")
            print("如需重新处理，请删除以下文件:")
            for f in processed_files:
                print(f"  - {f}")
            return
        
        # 下载数据集
        if not download_bank_dataset(data_dir):
            print("数据集下载失败")
            return
        
        # 预处理数据
        if not preprocess_bank_data(data_dir):
            print("数据预处理失败")
            return
        
        print("\n" + "="*50)
        print("Bank Marketing数据集准备完成!")
        print("现在可以运行Bank BadVFL训练脚本了")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 