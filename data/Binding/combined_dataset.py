import pandas as pd
import ast
import os

def parse_mutdict(mut_str):
    """将 "{'A': 'P11C', 'B': ''}" 转换为 "PA11C" """
    if pd.isna(mut_str): return ""
    try:
        mut_dict = ast.literal_eval(mut_str)
        res = []
        for chain, muts in mut_dict.items():
            if muts:
                for m in muts.split(','):
                    m = m.strip()
                    if len(m) >= 3:
                        wt, pos, mut_aa = m[0], m[1:-1], m[-1]
                        res.append(f"{wt}{chain}{pos}{mut_aa}")
        return ",".join(res)
    except Exception:
        return ""

def merge_all_bindinggym(master_csv_path, sub_csv_dir, pdb_base_dir, output_csv):
    print("="*60)
    print("🔍 [DEBUG] 路径配置检查")
    print(f"📍 当前工作目录: {os.getcwd()}")
    print(f"📍 主表文件路径: {os.path.abspath(master_csv_path)}")
    print(f"📍 子集存放目录: {os.path.abspath(sub_csv_dir)}")
    print(f"📍 PDB存放目录 : {os.path.abspath(pdb_base_dir)}")
    print("="*60 + "\n")

    if not os.path.exists(master_csv_path):
        print(f"❌ [致命错误] 找不到主表文件: {master_csv_path}")
        return

    master_df = pd.read_csv(master_csv_path)
    all_data = []

    for index, row in master_df.iterrows():
        complex_id = row.get('POI', f"Unknown_{index}")
        sub_csv_name = row.get('DMS_filename')
        pdb_file = row.get('pdb_file')

        print(f"--- 正在处理 POI: {complex_id} ---")

        # 1. 检查子集文件名
        if pd.isna(sub_csv_name):
            print(f"⚠️ [跳过] DMS_filename 字段为空。")
            continue

        # 确保文件名带有 .csv 后缀
        sub_csv_name = str(sub_csv_name).strip()
        if not sub_csv_name.endswith('.csv'):
            sub_csv_name += '.csv'

        # 🚨 核心修改：将子集所在目录与文件名拼接
        sub_csv_path = os.path.join(sub_csv_dir, sub_csv_name)

        # 2. 检查子集文件是否存在
        if not os.path.exists(sub_csv_path):
            print(f"❌ [跳过] 找不到子集文件，试图读取的路径是: {os.path.abspath(sub_csv_path)}")
            continue

        # 3. 开始处理子集
        try:
            sub_df = pd.read_csv(sub_csv_path)
            print(f"✅ [成功] 找到并读取子集，共 {len(sub_df)} 行数据。")

            # 填充标准化字段
            pdb_path = os.path.join(pdb_base_dir, str(pdb_file).strip())
            sub_df['pdb_path'] = pdb_path
            sub_df['complex_id'] = complex_id
            sub_df['source'] = sub_csv_name.split('.')[0]  # 去掉 .csv 后缀作为来源标识
            
            # 提取突变和分数
            if 'mutant' in sub_df.columns:
                sub_df['mutstr'] = sub_df['mutant'].apply(parse_mutdict)
            else:
                print(f"❌ [跳过] 缺少 'mutant' 列。现有列: {sub_df.columns.tolist()}")
                continue

            if 'DMS_score' in sub_df.columns:
                sub_df['dG_bind'] = sub_df['DMS_score']
            else:
                print(f"❌ [跳过] 缺少 'DMS_score' 列。现有列: {sub_df.columns.tolist()}")
                continue
            
            # 提取受体和配体
            if 'Entity1_chains' in sub_df.columns:
                sub_df['receptor'] = sub_df['Entity1_chains']
            else:
                sub_df['receptor'] = ""
                print(f"⚠️ [警告] 缺少 'Entity1_chains' 列，模型可能无法区分 Target。")

            if 'Entity2_chains' in sub_df.columns:
                sub_df['ligand'] = sub_df['Entity2_chains']
            else:
                sub_df['ligand'] = ""
                print(f"⚠️ [警告] 缺少 'Entity2_chains' 列，模型可能无法区分 Binder。")
                
            # 筛选我们需要的列并剔除没有分数的行
            clean_cols = ['complex_id', 'pdb_path', 'receptor', 'ligand', 'mutstr', 'dG_bind', 'source']
            df_to_append = sub_df[clean_cols].dropna(subset=['dG_bind'])
            
            print(f"🌟 [完成] {complex_id} 提取了 {len(df_to_append)} 条有效突变数据。\n")
            all_data.append(df_to_append)

        except Exception as e:
            print(f"❌ [报错] 处理 {sub_csv_path} 时发生异常: {e}\n")

    # 4. 合并保存
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print("="*60)
        print(f"🎉 完美收官！成功整合了 {len(final_df)} 条训练数据。")
        print(f"📁 最终合并文件已保存至: {os.path.abspath(output_csv)}")
        print("="*60)
    else:
        print("\n⚠️ 最终没有提取到任何有效数据，请检查上面的报错日志以定位问题！")

if __name__ == "__main__":
    # ==========================================================
    # 🚨 请在这里配置你服务器上的真实路径 🚨
    # ==========================================================
    
    # 1. 总控表 BindingGYM.csv 的路径
    MASTER_CSV = "/lustre/home/kwchen/dataset/BindingGYM/input/BindingGYM.csv"          
    
    # 2. 存放所有子集 (如 PSD95_Tm2F_1BE9.csv) 的文件夹路径
    SUBSET_CSV_DIR = "/lustre/home/kwchen/git/Stab2PPB/data/Binding/trainset"  
    SUBSET_CSV_DIR_1 = "/lustre/home/kwchen/git/Stab2PPB/data/Binding/testset" 
    SUBSET_CSV_DIR_2 = "/lustre/home/kwchen/git/Stab2PPB/data/Binding/validset"  
    
    # 3. 存放所有对应 PDB 结构文件的文件夹路径
    PDB_BASE_DIR = "/lustre/home/kwchen/dataset/BindingGYM/input/structures"                 
    
    # 4. 最终生成的合并训练集名字
    OUTPUT_FILE = "BindingGYM_Master_Train.csv"
    OUTPUT_FILE_1 = "BindingGYM_Master_Test.csv"
    OUTPUT_FILE_2 = "BindingGYM_Master_Valid.csv"
    
    # 开始执行
    merge_all_bindinggym(MASTER_CSV, SUBSET_CSV_DIR, PDB_BASE_DIR, OUTPUT_FILE)
    merge_all_bindinggym(MASTER_CSV, SUBSET_CSV_DIR_2, PDB_BASE_DIR, OUTPUT_FILE_2)