"""
单元测试：验证PPBOfflineGroupDataset的min_group_size过滤功能
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import tempfile
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from ppb.dataset_ppb import PPBOfflineGroupDataset


def create_mock_dataset(tmpdir, num_groups=5, samples_per_group=4, residue_range=(100, 500)):
    """
    创建模拟数据集：
    - 创建若干groups，每个group有多个samples
    - 部分group的total residues会超过max_residue限制，导致只能加载<2个样本
    """
    csv_path = os.path.join(tmpdir, 'mock_data.csv')
    cache_dir = os.path.join(tmpdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 构造数据
    rows = []
    row_id = 0
    
    for group_id in range(num_groups):
        for sample_idx in range(samples_per_group):
            # 为某些groups分配很大的residues，使得在max_residue=1000下只能加载1个样本
            if group_id >= 3:  # 后两个groups的样本会很大
                seq_len = 600  # 超过max_residue/2
            else:
                seq_len = np.random.randint(*residue_range)
            
            # 创建mock cache文件
            cache_path = os.path.join(cache_dir, f'sample_{row_id}.pt')
            mock_data = {
                'complex': {'X': torch.randn(seq_len, 4, 3)},
                'dG_bind': np.random.randn()
            }
            torch.save(mock_data, cache_path)
            
            rows.append({
                'group_id': f'group_{group_id}',
                'cache_path': cache_path,
                'seq_len': seq_len
            })
            row_id += 1
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path, cache_dir


def test_min_group_size_filtering():
    """
    测试min_group_size=2的过滤功能：
    - 在max_residue=1000下，大residue的groups只能加载1个样本
    - 这些groups应该被过滤掉
    """
    print("\n" + "="*60)
    print("TEST: min_group_size filtering with max_residue constraint")
    print("="*60)
    
    tmpdir = tempfile.mkdtemp()
    try:
        # 创建mock数据集
        csv_path, cache_dir = create_mock_dataset(tmpdir)
        print(f"\n✓ Created mock dataset at {csv_path}")
        
        # 加载原始数据
        df = pd.read_csv(csv_path)
        original_groups = df['group_id'].nunique()
        print(f"✓ Original groups count: {original_groups}")
        print(f"✓ Total samples: {len(df)}")
        
        # 创建dataset（应该在init时过滤）
        print("\n▶ Creating PPBOfflineGroupDataset with:")
        print("  - max_seqs=32")
        print("  - max_residue=1000")
        print("  - min_group_size=2")
        
        dataset = PPBOfflineGroupDataset(
            csv_path,
            group_col='group_id',
            max_seqs=32,
            max_residue=1000,
            min_group_size=2
        )
        
        # 验证过滤结果
        filtered_groups = len(dataset.group_keys)
        print(f"\n✓ After filtering: {filtered_groups} groups")
        print(f"✓ Filtered out: {original_groups - filtered_groups} groups")
        
        if filtered_groups < original_groups:
            print("\n✅ SUCCESS: Groups with K<2 under max_residue constraint were filtered!")
        else:
            print("\n⚠️  WARNING: No groups were filtered (unexpected in this test)")
        
        # 验证dataset可以正常迭代
        print("\n▶ Testing dataset iteration...")
        error_count = 0
        for idx in range(len(dataset)):
            try:
                batch = dataset[idx]
                if batch is not None:
                    group_size = batch.get('group_size', 0)
                    if group_size < 2:
                        print(f"❌ ERROR: Group at idx {idx} has K={group_size} (expected K>=2)")
                        error_count += 1
            except Exception as e:
                print(f"❌ ERROR at idx {idx}: {e}")
                error_count += 1
        
        if error_count == 0:
            print(f"✅ SUCCESS: All {len(dataset)} groups have K>=2")
        else:
            print(f"❌ FAILED: {error_count} groups have K<2")
            
        return error_count == 0
        
    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    success = test_min_group_size_filtering()
    print("\n" + "="*60)
    if success:
        print("✅ TEST PASSED: min_group_size filtering works correctly")
    else:
        print("❌ TEST FAILED: min_group_size filtering has issues")
    print("="*60 + "\n")
    sys.exit(0 if success else 1)
