import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback  # 引入 traceback 模块打印详细报错

# 请确保引入你的 PDB 解析函数
from utils.protein_mpnn_utils import parse_PDB, tied_featurize 

def compute_knn_graph(X, mask, k=48):
    X_ca = X[:, :, 1, :]
    D = torch.cdist(X_ca, X_ca)
    mask_2D = mask.unsqueeze(-1) * mask.unsqueeze(1)
    D = D + (1.0 - mask_2D) * 1e6
    D.diagonal(dim1=1, dim2=2).fill_(1e6)
    actual_k = min(k, X.shape[1])
    _, E_idx = torch.topk(D, actual_k, dim=-1, largest=False)
    return E_idx

def process_single_row(args):
    """
    多进程的 Worker 函数：处理单行数据，返回 (索引, 缓存路径, 序列长度)
    """
    torch.set_num_threads(1) 
    
    idx, row_dict, output_dir, k = args
    pdb_id = row_dict.get('id', f"ppb_{idx}")
    save_path = os.path.abspath(os.path.join(output_dir, f"{pdb_id}.pt"))
    
    # 1. 断点续传处理：如果文件已存在，加载它以获取 seq_len
    if os.path.exists(save_path):
        try:
            cache_data = torch.load(save_path, weights_only=False)
            # 校验是否完整包含三个状态，如果不完整则重新生成
            if all(key in cache_data for key in ['complex', 'binder', 'target']):
                seq_len = cache_data['complex']['S'].shape[0]
                return idx, save_path, seq_len
        except Exception:
            pass # 如果文件损坏或不完整，跳过断点续传，重新生成
            
    # 2. 正常处理流程 (自动从复合物中拆分 binder 和 target)
    try:
        cache_data = {}
        
        # 获取 PDB 路径
        pdb_path = row_dict.get('pdb_path')
        if not pdb_path or pd.isna(pdb_path):
            raise ValueError("CSV 中缺少 pdb_path 列，无法读取 PDB 文件。")
            
        # 解析 CSV 中的 ligand 和 receptor 链 ID (处理诸如 "A, B" 这种格式)
        ligand_str = str(row_dict.get('ligand', ''))
        receptor_str = str(row_dict.get('receptor', ''))
        binder_chains = [c.strip() for c in ligand_str.split(',') if c.strip()]
        target_chains = [c.strip() for c in receptor_str.split(',') if c.strip()]
        
        if not binder_chains or not target_chains:
            raise ValueError(f"CSV 中缺少 ligand 或 receptor 链信息。当前 ligand: '{ligand_str}', receptor: '{receptor_str}'")

        # 定义我们要提取的三个状态及对应的链 (None 表示提取所有链)
        tasks = [
            ('complex', None),
            ('binder', binder_chains),
            ('target', target_chains)
        ]
        
        for state, chains in tasks:
            # 提取指定链的 PDB 信息
            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chains)
            if not pdb_dict_list:
                raise ValueError(f"无法从 PDB 提取 {state} 状态 (试图提取链: {chains})，请检查 PDB 文件中是否包含这些链。")
                
            # 1. 提取出返回的元组
            tied_out = tied_featurize(pdb_dict_list, device=torch.device('cpu'), chain_dict=None)
            
            # 2. 补全模型需要的所有特征！
            X = tied_out[0]
            S = tied_out[1]
            mask = tied_out[2]
            chain_M = tied_out[4]               # 【新增】获取 chain_M
            chain_encoding_all = tied_out[5]
            residue_idx = tied_out[12]
            
            # 3. 计算 KNN 图
            E_idx = compute_knn_graph(X, mask, k=k)
            
            # 4. 构建字典（严格对齐 models.py 里的 batch 键名）
            cache_data[state] = {
                'X': X.squeeze(0).clone(),             
                'aa': S.squeeze(0).clone(),             # 【修改】将 S 改名为 aa
                'mask': mask.squeeze(0).clone(),       
                'chain_M': chain_M.squeeze(0).clone(),  # 【新增】存入 chain_M
                'E_idx': E_idx.squeeze(0).clone(),     
                'residue_idx': residue_idx.squeeze(0).clone(),
                'chain_encoding_all': chain_encoding_all.squeeze(0).clone()
            }
        
        # 提取真实标签 dG (兼容列名为 dG 或 dG_bind 的情况)
        if 'dG' in row_dict and not pd.isna(row_dict['dG']):
            cache_data['dG_bind'] = torch.tensor(row_dict['dG'], dtype=torch.float32)
        elif 'dG_bind' in row_dict and not pd.isna(row_dict['dG_bind']):
            cache_data['dG_bind'] = torch.tensor(row_dict['dG_bind'], dtype=torch.float32)
            
        # 保存完整的数据
        torch.save(cache_data, save_path)
        
        # 提取 complex 的长度并返回
        seq_len = cache_data['complex']['aa'].shape[0]
        return idx, save_path, seq_len
        
    except Exception as e:
        # 当发生错误时，不要静默失败，直接将详细报错打印到终端
        print(f"\n❌ [错误] 处理第 {idx} 行 (ID: {pdb_id}) 时发生异常:")
        print(traceback.format_exc())
        return idx, None, 0

def process_and_cache_parallel(csv_path, output_dir, output_csv_path, k=48, num_workers=None):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到文件 {csv_path}，跳过处理。")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2) 
        
    print(f"🚀 Starting parallel offline processing for {csv_path}")
    print(f"⚙️ Using {num_workers} CPU cores...")
    
    tasks = [(idx, row.to_dict(), output_dir, k) for idx, row in df.iterrows()]
    
    results_dict = {}
    len_dict = {}
    
    with Pool(processes=num_workers) as pool:
        for idx, cache_path, seq_len in tqdm(pool.imap_unordered(process_single_row, tasks), total=len(tasks)):
            results_dict[idx] = cache_path
            len_dict[idx] = seq_len
            
    cache_paths = [results_dict[i] for i in range(len(df))]
    seq_lens = [len_dict[i] for i in range(len(df))]
    
    df['cache_path'] = cache_paths
    df['seq_len'] = seq_lens
    
    df_clean = df.dropna(subset=['cache_path'])
    df_clean.to_csv(output_csv_path, index=False)
    print(f"✅ Finished! Retained {len(df_clean)}/{len(df)} valid samples. Saved to {output_csv_path}\n")

if __name__ == '__main__':
    # 确保 cache 文件夹存在
    # os.makedirs("cache", exist_ok=True)
    os.makedirs("train_cache", exist_ok=True)
    os.makedirs("val_cache", exist_ok=True)
    os.makedirs("test_cache", exist_ok=True)

    process_and_cache_parallel("train.csv", "train_cache", "train_cached.csv")
    process_and_cache_parallel("val.csv", "val_cache", "val_cached.csv")
    process_and_cache_parallel("test.csv", "test_cache", "test_cached.csv")