import pandas as pd
import networkx as nx
from collections import defaultdict

def create_pairwise_5fold(clusters_tsv, benchmark_csv, output_csv, k_folds=5):
    print(f"🚀 开始执行平衡 {k_folds} 折划分 | 隔离策略: [复合物级别 (受体+配体必须同时同源) 脱钩]")
    
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
                # 使用 sorted 排序，完美兼容你说的“或者反过来”的情况
                # 无论谁是受体谁是配体，只要互作的两端同源，就被视为同一种复合物
                pair = tuple(sorted([lc, rc]))
                pdb_to_pairs[pdb_id].add(pair)

    # 3. 构建 PDB 关联图：仅当两个 PDB 具有完全相同的 (受体聚类, 配体聚类) 组合时，才不可分割
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

    # 5. 贪婪降序装箱
    block_weights.sort(key=lambda x: x['weight'], reverse=True)

    folds = {i: [] for i in range(k_folds)}
    fold_sizes = {i: 0 for i in range(k_folds)}

    for block in block_weights:
        smallest_fold = min(fold_sizes, key=fold_sizes.get)
        folds[smallest_fold].extend(block['pdbs'])
        fold_sizes[smallest_fold] += block['weight']

    # 6. 打印报告并保存
    print("\n📊 最终各折叠 (Fold) 数据行数分布:")
    total_rows = len(df)
    for f_id in range(k_folds):
        size = fold_sizes[f_id]
        print(f"  Fold {f_id}: {size} 行 (占比 {size/total_rows*100:.1f}%)")

    pdb_to_fold = {}
    for fold_idx, pdbs in folds.items():
        for pdb in pdbs:
            pdb_to_fold[pdb] = fold_idx

    df['fold'] = df['pdb'].map(pdb_to_fold)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 均衡五折划分已完成！文件已保存至: {output_csv}")

if __name__ == "__main__":
    create_pairwise_5fold(
        clusters_tsv='clusters/clusters.tsv', 
        benchmark_csv='../benchmark.csv', 
        output_csv='../benchmark_5fold_pairwise.csv'
    )



