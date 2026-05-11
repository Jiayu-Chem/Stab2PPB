# import os
# import torch
# import numpy as np
# import pandas as pd
# from Bio.PDB import PDBParser
# from Bio.SeqUtils import seq1

# ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
# AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
# PAD_IDX = 20

# def compute_knn_graph(X, mask, k=48):
#     X_ca = X[:, :, 1, :]
#     D = torch.cdist(X_ca, X_ca)
#     mask_2D = mask.unsqueeze(-1) * mask.unsqueeze(1)
#     D = D + (1.0 - mask_2D) * 1e6
#     D.diagonal(dim1=1, dim2=2).fill_(1e6)
#     actual_k = min(k, X.shape[1])
#     _, E_idx = torch.topk(D, actual_k, dim=-1, largest=False)
#     return E_idx

# def extract_wt_features(pdb_path, target_chains):
#     parser = PDBParser(QUIET=True)
#     try: model = parser.get_structure('protein', pdb_path)[0]
#     except Exception as e: 
#         print(f"      [!] Bio.PDB 解析报错: {e}")
#         return None

#     X, aa, residue_idx, chain_encoding_all = [], [], [], []
#     chain_idx, res_count, current_idx = 1, 1, 0
#     pos_mapping = {}  

#     for chain in model:
#         ch_id = chain.get_id()
#         if target_chains and ch_id not in target_chains: continue
            
#         for residue in chain:
#             if residue.id[0] != ' ': continue
#             res_num = str(residue.id[1])
#             single_aa = seq1(residue.get_resname(), custom_map={"UNK": "X"})
#             if not single_aa: continue
            
#             coords = []
#             for atom_name in ['N', 'CA', 'C', 'O']:
#                 coords.append(residue[atom_name].get_coord() if atom_name in residue else np.full(3, np.nan))
            
#             X.append(coords)
#             aa.append(AA_DICT.get(single_aa, PAD_IDX))
#             residue_idx.append(res_count)
#             chain_encoding_all.append(chain_idx)
#             pos_mapping[(ch_id, res_num)] = current_idx 
            
#             current_idx += 1
#             res_count += 1
#         chain_idx += 1
        
#     if len(X) == 0: 
#         print(f"      [!] 未从 PDB 中提取到任何有效坐标，请检查链名是否真的存在于 PDB 中: {target_chains}")
#         return None
    
#     X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
#     valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()

#     # 3. 计算 KNN 图
#     E_idx = compute_knn_graph(X, valid_mask, k=48)
    
#     return {
#         'X': torch.nan_to_num(X_tensor, nan=0.0), 
#         'aa': torch.tensor(aa, dtype=torch.long), 
#         'mask': valid_mask,
#         'chain_M': torch.ones(len(aa), dtype=torch.float32), 
#         'E_idx': E_idx,
#         'residue_idx': torch.tensor(residue_idx, dtype=torch.long),
#         'chain_encoding_all': torch.tensor(chain_encoding_all, dtype=torch.long),
#         'pos_mapping': pos_mapping
#     }
#     # 参考：
#     """
#     cache_data[state] = {
#                 'X': X.squeeze(0).clone(),             
#                 'aa': S.squeeze(0).clone(),             # 【修改】将 S 改名为 aa
#                 'mask': mask.squeeze(0).clone(),       
#                 'chain_M': chain_M.squeeze(0).clone(),  # 【新增】存入 chain_M
#                 'E_idx': E_idx.squeeze(0).clone(),     
#                 'residue_idx': residue_idx.squeeze(0).clone(),
#                 'chain_encoding_all': chain_encoding_all.squeeze(0).clone()
#             }
#     """

# def build_wt_caches(master_csv_path, pdb_base_dir="./pdbs/", out_dir="./wt_caches/", csv_dir="./csvs/"):
#     os.makedirs(out_dir, exist_ok=True)
#     master_df = pd.read_csv(master_csv_path)

#     print(f"🔍 脚本当前工作目录: {os.getcwd()}")
#     print(f"🔍 PDB 文件夹设定为: {os.path.abspath(pdb_base_dir)}\n")

#     for _, row in master_df.iterrows():
#         complex_id = row['POI']
#         pdb_path = os.path.join(pdb_base_dir, str(row['pdb_file']).strip())
#         sub_csv_name = row['DMS_filename']
        
#         print(f"--- 正在处理 {complex_id} ---")
        
#         # 1. 检查子集 CSV 路径
#         if pd.isna(sub_csv_name):
#             print(f"❌ 跳过: DMS_filename 字段为空。")
#             continue
            
#         sub_csv_path = os.path.join(csv_dir, str(sub_csv_name).strip())
#         # 如果 CSV 表格没有以 .csv 结尾，给它加上 (以防表格里写的是 4D5_HER2_fitness_1N8Z)
#         if not sub_csv_path.endswith('.csv'):
#             sub_csv_path += '.csv'
            
#         if not os.path.exists(sub_csv_path):
#             print(f"❌ 跳过: 找不到子集文件 '{sub_csv_path}'。请确认它是否在这个文件夹里！")
#             continue

#         # 2. 检查子集内容
#         try:
#             sub_df = pd.read_csv(sub_csv_path, nrows=1)
#             if 'Entity1_chains' not in sub_df.columns or 'Entity2_chains' not in sub_df.columns:
#                 print(f"❌ 跳过: 在 '{sub_csv_path}' 中找不到 'Entity1_chains' 或 'Entity2_chains'。现有表头: {sub_df.columns.tolist()}")
#                 continue
                
#             receptor = str(sub_df['Entity1_chains'].iloc[0]).split(',')
#             ligand = str(sub_df['Entity2_chains'].iloc[0]).split(',')
#             receptor = [c.strip() for c in receptor if c.strip()]
#             ligand = [c.strip() for c in ligand if c.strip()]
#             print(f"✅ 获取到链信息 -> 受体(Target): {receptor}, 配体(Binder): {ligand}")
#         except Exception as e:
#             print(f"❌ 跳过: 尝试读取 '{sub_csv_path}' 时发生异常: {e}")
#             continue

#         # 3. 检查 PDB 文件是否存在
#         if not os.path.exists(pdb_path):
#             print(f"❌ 跳过: 找不到 PDB 文件 '{pdb_path}'。")
#             continue

#         # 4. 提取 PDB 特征
#         print(f"📦 开始从 {pdb_path} 提取特征...")
#         dict_complex = extract_wt_features(pdb_path, ligand + receptor)
#         dict_binder = extract_wt_features(pdb_path, ligand)
#         dict_target = extract_wt_features(pdb_path, receptor)
        
#         if dict_complex and dict_binder and dict_target:
#             save_data = {'complex': dict_complex, 'binder': dict_binder, 'target': dict_target}
#             save_path = os.path.join(out_dir, f"{complex_id}_wt.pt")
#             torch.save(save_data, save_path)
#             print(f"✅ 成功! 特征已保存至 {save_path}\n")
#         else:
#             print(f"❌ 提取失败: PDB 中可能不包含指定的受体或配体链。\n")

# if __name__ == "__main__":
#     build_wt_caches(
#         "/lustre/home/kwchen/dataset/BindingGYM/input/BindingGYM.csv", 
#         pdb_base_dir="/lustre/home/kwchen/dataset/BindingGYM/input/structures", 
#         out_dir="./wt_caches/", 
#         csv_dir="/lustre/home/kwchen/dataset/BindingGYM/input/Binding_substitutions_DMS/"
#     )

import os
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_DICT = {a: i for i, a in enumerate(ALPHABET)}
PAD_IDX = 20

def extract_wt_features(pdb_path, target_chains, k=48):
    parser = PDBParser(QUIET=True)
    try: model = parser.get_structure('protein', pdb_path)[0]
    except Exception as e: 
        print(f"      [!] Bio.PDB 解析报错: {e}")
        return None

    X, aa, residue_idx, chain_encoding_all = [], [], [], []
    chain_idx, res_count, current_idx = 1, 1, 0
    pos_mapping = {}  

    for chain in model:
        ch_id = chain.get_id()
        if target_chains and ch_id not in target_chains: continue
            
        for residue in chain:
            if residue.id[0] != ' ': continue
            res_num = str(residue.id[1])
            single_aa = seq1(residue.get_resname(), custom_map={"UNK": "X"})
            if not single_aa: continue
            
            coords = []
            for atom_name in ['N', 'CA', 'C', 'O']:
                coords.append(residue[atom_name].get_coord() if atom_name in residue else np.full(3, np.nan))
            
            X.append(coords)
            aa.append(AA_DICT.get(single_aa, PAD_IDX))
            residue_idx.append(res_count)
            chain_encoding_all.append(chain_idx)
            pos_mapping[(ch_id, res_num)] = current_idx 
            
            current_idx += 1
            res_count += 1
        chain_idx += 1
        
    if len(X) == 0: 
        return None
    
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    valid_mask = torch.isfinite(X_tensor[:, 0, 0]).float()
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    # ==========================================
    # 🌟 [修复点]: 增加 KNN 图计算，补上缺失的 E_idx
    # ==========================================
    X_ca = X_tensor[:, 1, :] # 提取所有 CA 原子的坐标
    D = torch.cdist(X_ca, X_ca) # 计算所有原子两两之间的欧式距离矩阵
    
    # 屏蔽无效的原子坐标
    mask_2D = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(0)
    D = D + (1.0 - mask_2D) * 1e6
    D.diagonal().fill_(1e6) # 屏蔽自身节点
    
    # 获取最近的 K 个邻居 (默认 48，如果序列太短则取实际长度)
    actual_k = min(k, X_tensor.shape[0])
    _, E_idx = torch.topk(D, actual_k, dim=-1, largest=False)
    # ==========================================

    return {
        'X': X_tensor, 
        'aa': torch.tensor(aa, dtype=torch.long), 
        'mask': valid_mask,
        'chain_M': torch.ones(len(aa), dtype=torch.float32), 
        'residue_idx': torch.tensor(residue_idx, dtype=torch.long),
        'chain_encoding_all': torch.tensor(chain_encoding_all, dtype=torch.long),
        'pos_mapping': pos_mapping,
        'E_idx': E_idx  # 🌟 将计算好的 KNN 图邻居索引存入缓存字典！
    }

def build_wt_caches(master_csv_path, pdb_base_dir="./pdbs/", out_dir="./wt_caches/", csv_dir="./csvs/"):
    os.makedirs(out_dir, exist_ok=True)
    master_df = pd.read_csv(master_csv_path)

    for _, row in master_df.iterrows():
        complex_id = row['POI']
        pdb_path = os.path.join(pdb_base_dir, str(row['pdb_file']).strip())
        sub_csv_name = row['DMS_filename']
        
        if pd.isna(sub_csv_name): continue
        sub_csv_path = str(sub_csv_name).strip()
        if not sub_csv_path.endswith('.csv'): sub_csv_path += '.csv'
        sub_csv_path = os.path.join(csv_dir, sub_csv_path)
        if not os.path.exists(sub_csv_path): continue

        try:
            sub_df = pd.read_csv(sub_csv_path, nrows=1)
            receptor = [c.strip() for c in str(sub_df['Entity1_chains'].iloc[0]).split(',') if c.strip()]
            ligand = [c.strip() for c in str(sub_df['Entity2_chains'].iloc[0]).split(',') if c.strip()]
        except Exception: continue

        if not os.path.exists(pdb_path): continue

        print(f"📦 正在提取并计算 {complex_id} 的 KNN图缓存 ...")
        dict_complex = extract_wt_features(pdb_path, ligand + receptor)
        dict_binder = extract_wt_features(pdb_path, ligand)
        dict_target = extract_wt_features(pdb_path, receptor)
        
        if dict_complex and dict_binder and dict_target:
            save_data = {'complex': dict_complex, 'binder': dict_binder, 'target': dict_target}
            save_path = os.path.join(out_dir, f"{complex_id}_wt.pt")
            torch.save(save_data, save_path)

if __name__ == "__main__":
    # 记得将这里的路径修改为你的真实路径
    build_wt_caches(
        "/lustre/home/kwchen/dataset/BindingGYM/input/BindingGYM.csv", 
        pdb_base_dir="/lustre/home/kwchen/dataset/BindingGYM/input/structures", 
        out_dir="./wt_caches/", 
        csv_dir="/lustre/home/kwchen/dataset/BindingGYM/input/Binding_substitutions_DMS/"
    )