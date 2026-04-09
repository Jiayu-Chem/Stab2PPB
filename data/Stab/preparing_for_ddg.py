import pandas as pd
import os
import numpy as np
import argparse

def prepare_simplified_csv(input_csv, output_csv):
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 过滤掉插入和缺失，但【保留双突变 (包含 ':')】以及【野生型 ('wt')】
    if 'mut_type' in df.columns:
        df = df[~df['mut_type'].str.contains("ins|del", na=False)]
    
    # 获取 PDB 的绝对路径
    df['WT_name_clean'] = df['WT_name'].str.split('.pdb').str[0].str.replace("|", ":")
    
    # 提取序列
    df['seq'] = df['aa_seq']
    
    # 提取绝对能量 dG。如果遇到 '-' 或者缺失，pd.to_numeric 会自动将其转为 NaN
    df['dG'] = pd.to_numeric(df['dG_ML'], errors='coerce')
    
    # 选取我们需要的列
    final_df = df[['PDB_path', 'seq', 'dG', 'pTM']]
    
    # 过滤掉没有序列的异常行
    final_df = final_df.dropna(subset=['seq'])
    
    final_df.to_csv(output_csv, index=False)
    
    valid_dg = final_df['dG'].notna().sum()
    nan_dg = final_df['dG'].isna().sum()
    
    print(f"✅ Saved to {output_csv}")
    print(f"   - Total sequences    : {len(final_df)}")
    print(f"   - With valid dG      : {valid_dg}")
    print(f"   - Missing dG (NaNs)  : {nan_dg} (Will be safely masked during training)")

# 使用示例：
# prepare_simplified_csv("data_all/training/mega_train.csv", "./data/Stab/megascale_train_flat.csv", "/绝对路径/指向/pdbs/")
# prepare_simplified_csv("data_all/training/mega_val.csv", "./data/Stab/megascale_val_flat.csv", "/绝对路径/指向/pdbs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare simplified CSV for DDG training")
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Path to the original CSV file")
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="Path to save the simplified CSV file")
    
    args = parser.parse_args()
    prepare_simplified_csv(args.input_csv, args.output_csv)