#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix encoding issues by replacing Chinese characters with English
"""

import re

def fix_encoding():
    # Read the file
    try:
        with open('train_bank_villain_with_inference.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        with open('train_bank_villain_with_inference.py', 'r', encoding='gbk', errors='ignore') as f:
            content = f.read()

    # Replace Chinese characters with English equivalents
    replacements = {
        # Print statements and comments
        '加载': 'Loading',
        '数据集': 'dataset', 
        '仅真实数据': 'real data only',
        '发现预处理的银行营销数据，正在加载': 'Found preprocessed bank marketing data, loading',
        '成功加载预处理数据': 'Successfully loaded preprocessed data',
        '样本': 'samples',
        '特征': 'features', 
        '预处理数据不存在，尝试加载原始CSV数据': 'Preprocessed data not found, trying to load original CSV data',
        '找到CSV文件': 'Found CSV file',
        '正在加载和预处理CSV数据': 'Loading and preprocessing CSV data',
        '原始数据': 'Original data',
        '列': 'columns',
        '特征列数': 'Feature columns',
        '标签分布': 'Label distribution',
        '分类特征': 'Categorical features',
        '标签编码': 'Label encoding',
        '标准化特征': 'Standardizing features',
        'CSV数据预处理完成': 'CSV data preprocessing completed',
        '划分训练/测试集': 'Splitting train/test sets',
        '训练集': 'Training set',
        '测试集': 'Test set',
        '预处理数据已保存，下次加载将更快': 'Preprocessed data saved, next loading will be faster',
        '保存预处理数据失败': 'Failed to save preprocessed data',
        '数据质量验证通过': 'Data quality validation passed',
        '参与方': 'Party',
        '特征范围': 'feature range',
        '维': 'dims',
        '最终数据': 'Final data',
        '数据目录': 'Data directory',
        '创建Bank Marketing模型，总特征维度': 'Creating Bank Marketing models, total feature dimension',
        '创建恶意底部模型': 'Creating malicious bottom model',
        '输入维度': 'input_dim',
        '输出维度': 'output_dim',
        '创建正常底部模型': 'Creating normal bottom model', 
        '创建顶部模型': 'Creating top model',
        'VILLAIN触发器已设置到参与方': 'VILLAIN trigger set to party',
        '模型创建完成': 'Model creation completed',
        '预热阶段': 'Warmup phase',
        '启用后门': 'Enable backdoor',
        '后门权重': 'Backdoor weight',
        '梯度收集': 'Gradient collection',
        '批次': 'Batch',
        '历史样本数': 'Historical sample count',
        '标签推断在批次.*初始化成功': 'Label inference initialized successfully at batch',
        '轮数': 'epochs',
        '批次大小': 'batch size',
        '结果': 'Results',
        '保存最佳模型': 'Saving best model',
        '早停触发': 'Early stopping triggered',
        '最佳模型在': 'Best model at',
        '训练完成': 'Training completed',
        '最佳模型': 'Best model',
        # More specific patterns
        'Epoch.*预热阶段.*启用后门.*后门权重': lambda m: m.group(0).replace('预热阶段', 'Warmup phase').replace('启用后门', 'Enable backdoor').replace('后门权重', 'Backdoor weight'),
    }

    # Apply replacements
    for chinese, english in replacements.items():
        if callable(english):
            content = re.sub(chinese, english, content)
        else:
            content = content.replace(chinese, english)

    # Remove any remaining problematic characters
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)
    
    # Fix any broken syntax
    content = re.sub(r'\s+', ' ', content)
    content = content.replace('( ', '(').replace(' )', ')')

    # Save the file with proper encoding
    with open('train_bank_villain_with_inference.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print('Encoding issues fixed successfully!')

if __name__ == '__main__':
    fix_encoding() 