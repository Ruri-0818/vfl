# Bank Marketing 数据集获取与使用指南

## 数据集简介

Bank Marketing数据集是来自UCI机器学习仓库的经典数据集，包含葡萄牙银行营销活动的数据。该数据集用于预测客户是否会订阅银行的定期存款产品。

### 数据集特点
- **样本数量**: 45,211个样本
- **特征数量**: 16-20个特征（取决于版本）
- **任务类型**: 二分类（是否订阅定期存款）
- **数据类型**: 包含数值和分类特征
- **缺失值**: 无

## 获取数据集的方法

### 方法1：使用本项目的自动化脚本（推荐）

```bash
# 1. 首先安装所需依赖
pip install ucimlrepo pandas scikit-learn torch

# 2. 运行数据准备脚本
cd WithInference
python prepare_bank_data.py
```

这个脚本会：
- 自动下载Bank Marketing数据集
- 预处理数据（编码分类特征、标准化数值特征）
- 划分训练集和测试集
- 保存为适合PyTorch使用的格式

### 方法2：使用UCI官方Python包

```bash
# 安装UCI官方包
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo 

# 获取Bank Marketing数据集
bank_marketing = fetch_ucirepo(id=222) 

# 获取特征和标签数据
X = bank_marketing.data.features 
y = bank_marketing.data.targets 

# 查看数据集信息
print(bank_marketing.metadata) 
print(bank_marketing.variables)
```

### 方法3：手动下载

1. 访问官方链接：https://archive.ics.uci.edu/dataset/222/bank+marketing
2. 点击"Download"按钮下载ZIP文件（约999.8 KB）
3. 解压文件，会得到以下几个CSV文件：

   - `bank.csv` - 小数据集（4,521个样本，17个特征）
   - `bank-full.csv` - 完整数据集（45,211个样本，17个特征）
   - `bank-additional.csv` - 新版小数据集（4,119个样本，20个特征）
   - `bank-additional-full.csv` - 新版完整数据集（41,188个样本，20个特征）

4. 将数据文件放置在 `./data/bank/` 目录下

### 方法4：从GitHub镜像下载

```python
import pandas as pd

# 从GitHub镜像读取
url = "https://raw.githubusercontent.com/YingluDeng/UCI_Bank_Marketing_ML/main/bank.csv"
df = pd.read_csv(url, sep=';')  # 注意分隔符是分号
```

## 数据集特征说明

### 客户基本信息
1. **age** - 年龄（数值）
2. **job** - 职业类型（分类）
3. **marital** - 婚姻状况（分类）
4. **education** - 教育水平（分类）
5. **default** - 是否有信用违约（二元）
6. **balance** - 平均年余额，欧元（数值）
7. **housing** - 是否有住房贷款（二元）
8. **loan** - 是否有个人贷款（二元）

### 最后一次联系信息
9. **contact** - 联系方式（分类：手机/电话）
10. **day** - 最后联系日期（数值）
11. **month** - 最后联系月份（分类）
12. **duration** - 最后通话时长，秒（数值）

### 其他属性
13. **campaign** - 本次活动联系次数（数值）
14. **pdays** - 距离上次活动的天数（数值，-1表示未联系过）
15. **previous** - 之前活动的联系次数（数值）
16. **poutcome** - 之前活动的结果（分类）

### 目标变量
17. **y** - 是否订阅定期存款（二元：yes/no）

## 在VILLAIN攻击中使用

### 1. 准备数据

```bash
# 运行数据准备脚本
python prepare_bank_data.py
```

### 2. 训练VILLAIN攻击模型

使用优化后的参数：

```bash
python train_bank_villain_with_inference.py \
  --dataset BANK \
  --data-dir ./data/bank \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.002 \
  --trigger-size 0.08 \
  --trigger-magnitude 0.4 \
  --poison-budget 0.06 \
  --inference-weight 0.15 \
  --confidence-threshold 0.35 \
  --Ebkd 5 \
  --warmup-epochs 4 \
  --backdoor-weight 0.25 \
  --clean-loss-weight 1.2 \
  --early-stopping \
  --gpu 0
```

### 3. 参数说明

- `--trigger-size 0.08`: 触发器大小，针对表格数据优化
- `--trigger-magnitude 0.4`: 触发器强度，平衡隐蔽性和有效性
- `--poison-budget 0.06`: 毒化预算，保持较低以避免检测
- `--inference-weight 0.15`: 标签推断损失权重
- `--confidence-threshold 0.35`: 标签推断置信度阈值
- `--Ebkd 5`: 第5轮开始后门攻击
- `--warmup-epochs 4`: 前4轮专注于clean accuracy

## 预期结果

### 训练目标
- **Clean Accuracy**: ≥ 85%（干净样本准确率）
- **Inference Accuracy**: ≥ 70%（标签推断准确率）
- **Attack Success Rate**: ≥ 80%（后门攻击成功率）

### 垂直联邦学习设置
- **参与方数量**: 3个
- **特征分割**: 每个参与方获得约5-6个特征
- **恶意方**: 第1个参与方（party 1）
- **目标类**: 0（未订阅定期存款）

## 故障排除

### 常见问题

1. **数据下载失败**
   ```
   错误: 在线下载失败
   解决: 检查网络连接，或手动下载数据集
   ```

2. **UCI包安装失败**
   ```bash
   pip install --upgrade pip
   pip install ucimlrepo
   ```

3. **内存不足**
   ```
   降低batch_size或使用较小的数据集版本
   ```

4. **GPU内存不足**
   ```bash
   # 使用CPU训练
   python train_bank_villain_with_inference.py --gpu -1
   ```

### 数据验证

运行以下命令验证数据是否正确准备：

```python
from prepare_bank_data import load_bank_data

# 加载数据
X_train, X_test, y_train, y_test, metadata = load_bank_data('./data/bank')

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"特征数量: {metadata['feature_dim']}")
print(f"类别数量: {metadata['num_classes']}")
print(f"目标类别: {metadata['target_classes']}")
```

## 引用

如果您在研究中使用此数据集，请引用：

```
Moro, S., Rita, P., & Cortez, P. (2014). 
Bank Marketing [Dataset]. UCI Machine Learning Repository. 
https://doi.org/10.24432/C5K306.
```

## 许可证

Bank Marketing数据集使用Creative Commons Attribution 4.0 International (CC BY 4.0)许可证。

## 更多资源

- [UCI官方页面](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [原始论文](https://www.sciencedirect.com/science/article/pii/S0167923614000550)
- [Kaggle上的分析示例](https://www.kaggle.com/search?q=bank+marketing) 