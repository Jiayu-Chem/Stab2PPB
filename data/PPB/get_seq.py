import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

def extract_sequences_from_csv(csv_path, output_fasta):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 提取所有唯一的 PDB 路径和对应的 PDB ID
    # 假设你的 CSV 中有 'pdb' 和 'pdb_path' 列
    unique_pdbs = df[['pdb', 'pdb_path']].drop_duplicates().dropna()
    
    parser = PDBParser(QUIET=True)
    fasta_records = []
    
    print("Extracting sequences from PDB files...")
    for _, row in tqdm(unique_pdbs.iterrows(), total=len(unique_pdbs)):
        pdb_id = str(row['pdb']).strip()
        pdb_path = str(row['pdb_path']).strip()
        
        if not os.path.exists(pdb_path):
            print(f"Warning: File not found -> {pdb_path}")
            continue
            
        try:
            # 解析 PDB 结构
            structure = parser.get_structure(pdb_id, pdb_path)
            # 通常只取第一个 Model
            model = structure[0]
            
            for chain in model:
                chain_id = chain.get_id()
                # 提取该链中所有标准氨基酸的序列
                seq = []
                for residue in chain:
                    # 过滤掉水分子和杂原子 (Hetero atoms)
                    if residue.id[0] == ' ': 
                        res_name = residue.get_resname()
                        # 将三字母缩写转换为单字母，未知氨基酸用 'X' 表示
                        single_aa = seq1(res_name, custom_map={"UNK": "X"})
                        if single_aa:  # 确保不是空值
                            seq.append(single_aa)
                
                # 如果该链有序列，则加入记录
                if len(seq) > 0:
                    sequence_str = "".join(seq)
                    # FASTA Header 格式: >PDBID_CHAIN
                    header = f">{pdb_id}_{chain_id}"
                    fasta_records.append(f"{header}\n{sequence_str}")
                    
        except Exception as e:
            print(f"Error processing {pdb_id} at {pdb_path}: {e}")

    # 写入 FASTA 文件
    print(f"\nWriting {len(fasta_records)} chain sequences to {output_fasta}...")
    with open(output_fasta, 'w') as f:
        f.write("\n".join(fasta_records) + "\n")
    print("Done!")

if __name__ == "__main__":
    # 配置你的路径
    CSV_FILE = "benchmark.csv"          # 你的数据集表格
    OUTPUT_FASTA = "ppb_all_chains.fasta" # 导出的 FASTA 文件名
    
    extract_sequences_from_csv(CSV_FILE, OUTPUT_FASTA)