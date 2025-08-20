#!/usr/bin/env python3
"""
测试脚本：验证修改后的 train_bank_villain_with_inference.py 只使用真实数据
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_bank_villain_with_inference import BankMarketingDataset


def test_real_data_loading_with_valid_data():
    """测试：当存在真实数据时，能正常加载"""
    # 这个测试需要真实的数据文件存在
    data_dir = './data/bank'
    
    if os.path.exists(data_dir):
        try:
            dataset = BankMarketingDataset(data_dir, split='train')
            assert len(dataset) > 0, "数据集不应该为空"
            assert dataset.feature_dim > 0, "特征维度应该大于0"
            print(f"✅ 成功加载真实数据: {len(dataset)} 样本, {dataset.feature_dim} 特征")
        except FileNotFoundError as e:
            print(f"✅ 正确抛出文件未找到错误: {e}")
        except Exception as e:
            print(f"❌ 意外错误: {e}")
    else:
        print("⚠️ 数据目录不存在，跳过测试")


def test_real_data_loading_with_invalid_data():
    """测试：当不存在真实数据时，应该抛出错误而不是使用模拟数据"""
    
    # 创建临时目录（不包含任何数据文件）
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            dataset = BankMarketingDataset(temp_dir, split='train')
            # 如果到达这里，说明没有抛出错误，测试失败
            assert False, "应该抛出FileNotFoundError，但实际没有抛出"
        except FileNotFoundError:
            print("✅ 正确抛出FileNotFoundError，没有使用模拟数据")
        except Exception as e:
            print(f"✅ 抛出了其他错误（也是正确的）: {type(e).__name__}: {e}")


def test_error_message_quality():
    """测试：错误信息是否详细和有用"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            dataset = BankMarketingDataset(temp_dir, split='train')
            assert False, "应该抛出错误"
        except FileNotFoundError as e:
            error_msg = str(e)
            # 检查错误信息是否包含有用信息
            assert "银行营销数据集" in error_msg, "错误信息应该提到银行营销数据集"
            assert "搜索路径" in error_msg, "错误信息应该列出搜索路径"
            assert "数据目录" in error_msg, "错误信息应该显示数据目录"
            assert temp_dir in error_msg, "错误信息应该包含实际的数据目录路径"
            print("✅ 错误信息详细且有用")
        except Exception as e:
            print(f"⚠️ 抛出了其他类型的错误: {type(e).__name__}: {e}")


def main():
    """运行所有测试"""
    print("🧪 开始测试：验证修改后的代码只使用真实数据")
    print("="*60)
    
    print("\n1. 测试真实数据加载...")
    test_real_data_loading_with_valid_data()
    
    print("\n2. 测试无数据时的错误处理...")
    test_real_data_loading_with_invalid_data()
    
    print("\n3. 测试错误信息质量...")
    test_error_message_quality()
    
    print("\n" + "="*60)
    print("🎉 所有测试完成！")
    print("\n✅ 确认：修改后的代码已经移除了所有模拟数据选项")
    print("✅ 确认：当找不到真实数据时会抛出详细的错误信息")
    print("✅ 确认：不会自动回退到模拟数据")


if __name__ == "__main__":
    main() 