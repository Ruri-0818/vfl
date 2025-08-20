import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def test_bank_data_loading():
    """测试Bank Marketing数据集加载"""
    
    print("="*60)
    print("测试Bank Marketing数据集加载")
    print("="*60)
    
    # 检查数据文件路径
    data_path = './data/bank/bank-additional/bank-additional.csv'
    full_data_path = './data/bank/bank-additional/bank-additional-full.csv'
    
    print(f"检查数据文件...")
    print(f"小数据集路径: {data_path}")
    print(f"完整数据集路径: {full_data_path}")
    print(f"小数据集存在: {os.path.exists(data_path)}")
    print(f"完整数据集存在: {os.path.exists(full_data_path)}")
    
    if not os.path.exists(data_path):
        print("❌ 数据文件不存在！")
        return False
    
    # 加载数据
    print(f"\n正在加载数据...")
    try:
        # 注意：Bank Marketing数据集使用分号作为分隔符
        df = pd.read_csv(data_path, sep=';')
        print(f"✅ 数据加载成功！")
        
        # 显示数据信息
        print(f"\n数据集基本信息:")
        print(f"  形状: {df.shape}")
        print(f"  列数: {len(df.columns)}")
        print(f"  行数: {len(df)}")
        
        print(f"\n列名:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 显示目标变量分布
        target_col = 'y'
        if target_col in df.columns:
            print(f"\n目标变量 '{target_col}' 分布:")
            value_counts = df[target_col].value_counts()
            for value, count in value_counts.items():
                percentage = count / len(df) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
        
        # 显示前几行数据
        print(f"\n前5行数据预览:")
        print(df.head())
        
        # 检查数据类型
        print(f"\n数据类型:")
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if col == target_col:
                continue
            dtype = df[col].dtype
            if dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
                unique_count = df[col].nunique()
                print(f"  {col}: 分类 ({unique_count} 个唯一值)")
            else:
                numerical_cols.append(col)
                print(f"  {col}: 数值")
        
        print(f"\n特征类型统计:")
        print(f"  分类特征: {len(categorical_cols)} 个")
        print(f"  数值特征: {len(numerical_cols)} 个")
        print(f"  总特征数: {len(categorical_cols) + len(numerical_cols)} 个")
        
        # 检查缺失值
        print(f"\n缺失值检查:")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            print("  ✅ 没有缺失值")
        else:
            print("  缺失值统计:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = missing_count / len(df) * 100
                    print(f"    {col}: {missing_count} ({percentage:.1f}%)")
        
        # 测试预处理
        print(f"\n测试数据预处理...")
        
        # 分离特征和标签
        features = df.drop(columns=[target_col])
        labels = df[target_col]
        
        # 编码分类特征
        print("  编码分类特征...")
        processed_features = features.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            processed_features[col] = le.fit_transform(processed_features[col].astype(str))
            print(f"    {col}: {len(le.classes_)} 个类别")
        
        # 编码目标标签
        print("  编码目标标签...")
        label_encoder = LabelEncoder()
        processed_labels = label_encoder.fit_transform(labels)
        print(f"    目标类别: {label_encoder.classes_}")
        
        # 标准化数值特征
        print("  标准化数值特征...")
        scaler = StandardScaler()
        processed_features[numerical_cols] = scaler.fit_transform(processed_features[numerical_cols])
        
        # 转换为numpy数组
        X = processed_features.values.astype(np.float32)
        y = processed_labels.astype(np.int64)
        
        print(f"\n预处理结果:")
        print(f"  特征矩阵形状: {X.shape}")
        print(f"  标签向量形状: {y.shape}")
        print(f"  标签分布: {np.bincount(y)}")
        
        # 测试数据划分
        print(f"\n测试数据划分...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")
        print(f"  训练集标签分布: {np.bincount(y_train)}")
        print(f"  测试集标签分布: {np.bincount(y_test)}")
        
        print(f"\n✅ 数据预处理测试成功！")
        
        # 为VFL准备数据
        print(f"\n为VFL准备数据分割...")
        num_parties = 3
        feature_dim = X.shape[1]
        features_per_party = feature_dim // num_parties
        
        print(f"  总特征数: {feature_dim}")
        print(f"  参与方数量: {num_parties}")
        print(f"  每方特征数: {features_per_party}")
        
        for i in range(num_parties):
            start_idx = i * features_per_party
            if i == num_parties - 1:
                end_idx = feature_dim  # 最后一方获得剩余特征
            else:
                end_idx = (i + 1) * features_per_party
            
            party_features = X_train[:, start_idx:end_idx]
            print(f"  参与方 {i}: 特征范围 [{start_idx}:{end_idx}], 维度 {party_features.shape[1]}")
        
        print(f"\n✅ VFL数据分割测试成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False

def create_processed_data():
    """创建预处理后的数据文件供训练使用"""
    
    print(f"\n" + "="*60)
    print("创建预处理数据文件")
    print("="*60)
    
    # 数据路径
    data_path = './data/bank/bank-additional/bank-additional-full.csv'
    if not os.path.exists(data_path):
        data_path = './data/bank/bank-additional/bank-additional.csv'
    
    if not os.path.exists(data_path):
        print("❌ 找不到数据文件")
        return False
    
    # 加载数据
    print(f"从 {data_path} 加载数据...")
    df = pd.read_csv(data_path, sep=';')
    print(f"原始数据形状: {df.shape}")
    
    # 预处理
    target_col = 'y'
    features = df.drop(columns=[target_col])
    labels = df[target_col]
    
    # 分类特征编码
    categorical_columns = features.select_dtypes(include=['object']).columns
    numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"分类特征: {len(categorical_columns)} 个")
    print(f"数值特征: {len(numerical_columns)} 个")
    
    # 编码分类特征
    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
    
    # 编码标签
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # 标准化数值特征
    scaler = StandardScaler()
    features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
    
    # 转换数据类型
    X = features.values.astype(np.float32)
    y = labels_encoded.astype(np.int64)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建输出目录
    output_dir = './data/bank'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    print(f"保存预处理数据到 {output_dir}...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # 保存元数据
    metadata = {
        'feature_names': list(features.columns),
        'categorical_columns': list(categorical_columns),
        'numerical_columns': list(numerical_columns),
        'target_classes': list(label_encoder.classes_),
        'feature_dim': X.shape[1],
        'num_classes': len(label_encoder.classes_),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import pickle
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ 数据保存完成！")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"特征维度: {metadata['feature_dim']}")
    print(f"类别数: {metadata['num_classes']}")
    
    return True

if __name__ == '__main__':
    # 测试数据加载
    success = test_bank_data_loading()
    
    if success:
        # 创建预处理数据
        create_processed_data()
        
        print(f"\n" + "="*60)
        print("🎉 Bank Marketing数据集验证完成！")
        print("="*60)
        print("数据集已准备就绪，可以开始训练VILLAIN攻击")
        print()
        print("运行训练命令:")
        print("python train_bank_villain_with_inference.py \\")
        print("  --dataset BANK \\")
        print("  --data-dir ./data/bank \\")
        print("  --batch-size 64 \\")
        print("  --epochs 50 \\")
        print("  --lr 0.002 \\")
        print("  --trigger-size 0.08 \\")
        print("  --trigger-magnitude 0.4 \\")
        print("  --poison-budget 0.06 \\")
        print("  --inference-weight 0.15 \\")
        print("  --confidence-threshold 0.35 \\")
        print("  --Ebkd 5 \\")
        print("  --warmup-epochs 4 \\")
        print("  --early-stopping \\")
        print("  --gpu 0")
        
    else:
        print("❌ 数据验证失败，请检查数据文件") 