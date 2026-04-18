import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
            seq_len = cache_data['complex']['S'].shape[0]
            return idx, save_path, seq_len
        except Exception:
            pass # 如果文件损坏，跳过断点续传，重新生成
            
    # 2. 正常处理流程
    try:
        cache_data = {}
        for state, path_col in [('complex', 'complex_pdb'), ('binder', 'binder_pdb'), ('target', 'target_pdb')]:
            if pd.isna(row_dict.get(path_col)): continue
            
            pdb_dict_list = parse_PDB(row_dict[path_col])
            batched_data = tied_featurize(pdb_dict_list, device=torch.device('cpu'), chain_dict=None)
            
            X = batched_data['X']
            S = batched_data['S']
            mask = batched_data['mask']
            
            E_idx = compute_knn_graph(X, mask, k=k)
            
            cache_data[state] = {
                'X': X.squeeze(0).clone(),             
                'S': S.squeeze(0).clone(),             
                'mask': mask.squeeze(0).clone(),       
                'E_idx': E_idx.squeeze(0).clone(),     
                'residue_idx': batched_data['residue_idx'].squeeze(0).clone(),
                'chain_encoding_all': batched_data['chain_encoding_all'].squeeze(0).clone()
            }
        
        if 'dG_bind' in row_dict and not pd.isna(row_dict['dG_bind']):
            cache_data['dG_bind'] = torch.tensor(row_dict['dG_bind'], dtype=torch.float32)
            
        torch.save(cache_data, save_path)
        
        # 【修复】提取 complex 的长度并返回
        seq_len = cache_data['complex']['S'].shape[0]
        return idx, save_path, seq_len
        
    except Exception as e:
        # 失败则返回 None 路径和 0 长度
        return idx, None, 0

def process_and_cache_parallel(csv_path, output_dir, output_csv_path, k=48, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2) 
        
    print(f"🚀 Starting parallel offline processing for {csv_path}")
    print(f"⚙️ Using {num_workers} CPU cores...")
    
    tasks = [(idx, row.to_dict(), output_dir, k) for idx, row in df.iterrows()]
    
    # 使用两个字典分别记录路径和长度
    results_dict = {}
    len_dict = {}
    
    with Pool(processes=num_workers) as pool:
        # 【修复】接收 3 个返回值
        for idx, cache_path, seq_len in tqdm(pool.imap_unordered(process_single_row, tasks), total=len(tasks)):
            results_dict[idx] = cache_path
            len_dict[idx] = seq_len
            
    # 按照原 DataFrame 的索引顺序组装列表
    cache_paths = [results_dict[i] for i in range(len(df))]
    seq_lens = [len_dict[i] for i in range(len(df))]
    
    # 写入 DataFrame
    df['cache_path'] = cache_paths
    df['seq_len'] = seq_lens
    
    # 过滤掉失败的数据
    df_clean = df.dropna(subset=['cache_path'])
    df_clean.to_csv(output_csv_path, index=False)
    print(f"✅ Finished! Retained {len(df_clean)}/{len(df)} valid samples. Saved to {output_csv_path}\n")

if __name__ == '__main__':
    process_and_cache_parallel("train.csv", "cache", "train_cached.csv")
    process_and_cache_parallel("val.csv", "cache", "val_cached.csv")
    process_and_cache_parallel("test.csv", "cache", "test_cached.csv")