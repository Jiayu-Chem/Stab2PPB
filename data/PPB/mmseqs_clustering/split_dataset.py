import pandas as pd
import networkx as nx
from collections import defaultdict

def create_pairwise_split(clusters_tsv, benchmark_csv, 
                          train_csv='train.csv', 
                          val_csv='val.csv', 
                          test_csv='test.csv', 
                          train_ratio=0.80, val_ratio=0.15, test_ratio=0.05):
    # 校验比例之和是否为 1 (使用微小误差避免浮点数精度问题)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例总和必须为 1.0"
    
    print(f"🚀 开始执行 {train_ratio}:{val_ratio}:{test_ratio} 划分 | 隔离策略: [复合物级别 (受体+配体必须同时同源) 脱钩]")
    
    # 1. 解析 clusters.tsv 构建链到 Cluster ID 的映射
    chain_graph = nx.Graph()
    with open(clusters_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                chain_graph.add_edge(parts[0], parts[1])

    chain_to_cluster = {}
    for cluster_id, component in enumerate(nx.connected_components(chain_graph)):
        for chain in component:
            chain_to_cluster[chain] = cluster_id
    max_cluster_id = len(list(nx.connected_components(chain_graph)))

    def get_cluster(chain_name):
        nonlocal max_cluster_id
        if chain_name in chain_to_cluster:
            return chain_to_cluster[chain_name]
        else:
            max_cluster_id += 1
            chain_to_cluster[chain_name] = max_cluster_id
            return max_cluster_id

    # 2. 提取每个 PDB 的 (受体Cluster, 配体Cluster) 组合
    df = pd.read_csv(benchmark_csv)
    pdb_to_pairs = defaultdict(set)

    for idx, row in df.iterrows():
        pdb_id = row['pdb']
        
        ligands = [c.strip() for c in str(row['ligand']).split(',')] if pd.notna(row['ligand']) else []
        receptors = [c.strip() for c in str(row['receptor']).split(',')] if pd.notna(row['receptor']) else []
        
        lig_clusters = {get_cluster(f"{pdb_id}_{c}") for c in ligands}
        rec_clusters = {get_cluster(f"{pdb_id}_{c}") for c in receptors}
        
        # 记录所有的配体-受体聚类对
        for lc in lig_clusters:
            for rc in rec_clusters:
                # 使用 sorted 排序，完美兼容受体/配体反转的情况
                pair = tuple(sorted([lc, rc]))
                pdb_to_pairs[pdb_id].add(pair)

    # 3. 构建 PDB 关联图：仅当两个 PDB 具有完全相同的复合物聚类组合时，才不可分割
    pdb_graph = nx.Graph()
    pdb_graph.add_nodes_from(df['pdb'].unique())

    pair_to_pdbs = defaultdict(list)
    for pdb, pairs in pdb_to_pairs.items():
        for p in pairs:
            pair_to_pdbs[p].append(pdb)

    for pdbs in pair_to_pdbs.values():
        for i in range(len(pdbs)):
            for j in range(i + 1, len(pdbs)):
                pdb_graph.add_edge(pdbs[i], pdbs[j])

    # 4. 获取独立块并计算行数权重
    pdb_blocks = list(nx.connected_components(pdb_graph))
    pdb_row_counts = df['pdb'].value_counts().to_dict()
    
    block_weights = []
    for block in pdb_blocks:
        weight = sum(pdb_row_counts[pdb] for pdb in block)
        block_weights.append({'pdbs': list(block), 'weight': weight})
        
    print(f"🔍 成功将数据集拆解为 {len(block_weights)} 个相互独立的复合物块")

    # 5. 贪婪降序装箱 (按目标比例划分)
    block_weights.sort(key=lambda x: x['weight'], reverse=True)
    
    total_rows = len(df)
    targets = {
        'train': total_rows * train_ratio,
        'val': total_rows * val_ratio,
        'test': total_rows * test_ratio
    }
    
    splits = {'train': [], 'val': [], 'test': []}
    split_sizes = {'train': 0, 'val': 0, 'test': 0}

    for block in block_weights:
        # 寻找当前填充进度最落后的集合
        best_split = min(
            splits.keys(), 
            key=lambda s: split_sizes[s] / targets[s] if targets[s] > 0 else float('inf')
        )
        
        splits[best_split].extend(block['pdbs'])
        split_sizes[best_split] += block['weight']

    # 6. 打印报告
    print("\n📊 最终数据集划分行数分布:")
    for s_name in ['train', 'val', 'test']:
        size = split_sizes[s_name]
        target_perc = (targets[s_name] / total_rows) * 100
        actual_perc = (size / total_rows) * 100
        print(f"  {s_name.ljust(5)}: {size} 行 (实际占比 {actual_perc:.1f}% | 目标占比 {target_perc:.1f}%)")

    # 7. 根据 PDB 列表拆分 DataFrame 并分别保存为三个文件
    df_train = df[df['pdb'].isin(splits['train'])]
    df_val = df[df['pdb'].isin(splits['val'])]
    df_test = df[df['pdb'].isin(splits['test'])]

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print("\n✅ 数据集已成功拆分并保存为三个文件:")
    print(f"  - 训练集: {train_csv}")
    print(f"  - 验证集: {val_csv}")
    print(f"  - 测试集: {test_csv}")

if __name__ == "__main__":
    create_pairwise_split(
        clusters_tsv='clusters/clusters.tsv', 
        benchmark_csv='../benchmark.csv', 
        train_csv='../train.csv',
        val_csv='../val.csv',
        test_csv='../test.csv',
        train_ratio=0.80,   # 训练集 80%
        val_ratio=0.15,     # 验证集 15%
        test_ratio=0.05     # 测试集 5%
    )