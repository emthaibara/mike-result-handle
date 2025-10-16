import json
import os
import shutil
import threading
from os import path
import mikeio
import numpy as np
import pandas as pd
from tqdm import tqdm

def gen(
        case_id : int,
        base_location : str,
        dataset_location : str):
    """ 读取原始数据并重命名列名称,z.dfs0除外 """
    case_result_path = path.join(base_location, cases_dict[case_id]['path'], 'LHKHX.m21fm - Result Files')
    df_lhk = mikeio.read(path.join(case_result_path, 'lhk.dfs0')).to_dataframe()
    df_lhk.columns = [f'{col}_lhk' for col in df_lhk.columns]

    df_lhkhx = mikeio.read(path.join(case_result_path, 'lhkhx.dfs0')).to_dataframe()
    df_lhkhx.columns = [f'{col}_lhkhx' for col in df_lhkhx.columns]

    df_ygyj = mikeio.read(path.join(case_result_path, 'ygyj.dfs0')).to_dataframe()
    df_ygyj.columns = [f'{col}_ygyj' for col in df_ygyj.columns]

    df_z = mikeio.read(path.join(case_result_path, 'z.dfs0')).to_dataframe()


    """ 接下来合并四个df, 为新df列命名 """
    output_columns = ['q1_input', 'q2_input', 'q3_input', 'q1_output', 'q2_output', 'q3_output']

    cross_sections = [
        'YG63LHK: Surface elevation', 'YG62-1.5: Surface elevation', 'YG62-1: Surface elevation',
        'YG62-0.5: Surface elevation', 'YG62: Surface elevation', 'YG61-2.5: Surface elevation',
        'YG61-2: Surface elevation', 'YG61-1.5: Surface elevation', 'YG61-1: Surface elevation',
        'YG61-0.9HX: Surface elevation', 'YG61-0.75: Surface elevation', 'YG61-0.5: Surface elevation',
        'YG61-0.25: Surface elevation', 'YG61: Surface elevation', 'YG60-1.75: Surface elevation',
        'YG60-1.5: Surface elevation', 'YG60-1.25: Surface elevation', 'YG60-1: Surface elevation',
        'YG60-0.5: Surface elevation', 'YG60: Surface elevation', 'YG59-0.5: Surface elevation',
        'YG59-1: Surface elevation', 'YG59-2: Surface elevation', 'YG59: Surface elevation',
        'YG58-1: Surface elevation', 'YG58: Surface elevation', 'YG57-1: Surface elevation',
        'YG57-2: Surface elevation', 'YG57: Surface elevation', 'YG56-1: Surface elevation',
        'YG56: Surface elevation', 'YG55-1: Surface elevation', 'YG55: Surface elevation',
        'YG54-1: Surface elevation', 'YG54: Surface elevation', 'YG53-2: Surface elevation',
        'YG53-1: Surface elevation', 'YG53: Surface elevation', 'YG52-1: Surface elevation',
        'YG52: Surface elevation', 'YG51-1: Surface elevation', 'YG51: Surface elevation',
        'YG50-1: Surface elevation', 'YG50: Surface elevation', 'YG49-1: Surface elevation',
        'YG49: Surface elevation', 'YG48-1: Surface elevation', 'YG48: Surface elevation',
        'YG47-2: Surface elevation', 'YG47: Surface elevation', 'YG47-1: Surface elevation',
        'YG46: Surface elevation', 'YG46-1: Surface elevation', 'YG45: Surface elevation',
        'YG45-1: Surface elevation', 'YG44YGYJ: Surface elevation'
    ]

    all_columns = output_columns + cross_sections

    missing_cols = [col for col in cross_sections if col not in df_z.columns]
    if missing_cols:
        print(f"错误：在 {path.join(case_result_path, 'z.dfs0')} 中找不到以下列：{missing_cols}")
        return
    """ 合并 """
    master_df = df_lhk.merge(df_lhkhx, left_index=True, right_index=True, how='outer')
    master_df = master_df.merge(df_ygyj, left_index=True, right_index=True, how='outer')
    master_df = master_df.merge(df_z, left_index=True, right_index=True, how='outer')

    output_df = pd.DataFrame(index=master_df.index, columns=all_columns)
    """
    Q1输入输出均为正数，保留1位小数；
    Q2输入输出统一，抽水为负，发电为正，保留1位小数，不抽不发设为0；
    Q3输入输出均为负数，保留1位小数。
    沿程水位输出保留3位小数。
    """
    q1_origin = cases_dict[case_id]['q1-flow_rate']
    q2_origin = cases_dict[case_id]['q2-flow_rate']
    q3_origin = cases_dict[case_id]['q3-flow_rate']
    output_df['q1_input'] = q1_origin
    output_df['q2_input'] = q2_origin
    output_df['q3_input'] = q3_origin

    output_df['q1_output'] = master_df[df_lhk.columns[0]].abs().round(1)

    if q2_origin == 0:
        output_df['q2_output'] = 0.0
    else:
        q2_sim = master_df[df_lhkhx.columns[0]].abs().round(1)
        if q2_origin < 0:
            output_df['q2_output'] = -q2_sim
        elif q2_origin > 0:
            output_df['q2_output'] = q2_sim

    output_df['q3_output'] = -master_df[df_ygyj.columns[0]].abs().round(1)

    for section_name in cross_sections:
        output_df[section_name] = master_df[section_name].round(3)

    output_df.to_csv(path.join(dataset_location, f'{case_id}.csv'), index=True, index_label='time', encoding='utf-8')

def move_only_csv_files(source_folder, destination_folder):
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        if os.path.isfile(source_path) and item.endswith('.csv'):
            shutil.move(source_path, destination_path)
    print("所有 .csv 文件移动完成。")
    return True

""" pump cases需要去掉第一个步长的数据,即为2023-01-01 09:00:00的数据 """
def clip(start_id : int,
         end_id : int,
         location : str):
    for cid in tqdm(range(start_id, end_id + 1),
                    desc='正在批量截取数据集的第一行数据'):
        df = pd.read_csv(path.join(location, f'{cid}.csv'), encoding='utf-8')
        df['time'] = pd.to_datetime(df['time'])
        df_filtered = df[df['time'] != '2023-01-01 09:00:00']
        df_filtered.to_csv(path.join(location, f'{cid}.csv'), index=False)

cases_location = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\cases.json'
cases_dict = dict()
def batch_gen_dataset(start_id : int,
                      end_id : int,
                      base_location : str,
                      dataset_location : str):
    for case_id in tqdm(range(start_id, end_id + 1),
                        desc=f'正在批量生成数据集【编号{start_id}-{end_id}】'):
        gen(case_id, base_location, dataset_location)

def init():
    cases = json.load(open(cases_location, 'r', encoding='utf-8'))
    cases_list = cases["cases"]
    global cases_dict
    cases_dict = {item['case_id']: item for item in cases_list}

def section_matrix(base_location):
    section_info_df = pd.read_csv(os.path.join(base_location, 'sections-info.csv'))
    section_distance_data = []
    section_point_data = []
    for row in section_info_df.itertuples(index=False):
        name, x, y, d = row[0], row[1], row[2], row[3]
        section_distance_data.append((name, d))
        section_point_data.append((name, x, y))
    # ====================================================================
    # I. 沿程距离矩阵 (|D - D^T|)
    # ====================================================================
    """ 两个断面点与两河口水坝的距离（沿程距离）做差取绝对值 【以下一个是km单位的版本，一个是米为单位的版本】"""
    index_labels = [f"P{i + 1} ({name})" for i, (name, d) in enumerate(section_distance_data)]
    distances_km = np.array([d for name, d in section_distance_data])
    raw_distance_matrix_km = np.abs(distances_km[:, np.newaxis] - distances_km)
    """ m """
    final_matrix_dist_km_w = apply_inverse_and_direction(raw_distance_matrix_km, dtype_conversion=1000)
    df_km_w = pd.DataFrame(final_matrix_dist_km_w, index=index_labels, columns=index_labels)
    csv_file_km_w = path.join(base_location, 'dataset','section_distance_along_the_way_m.csv')
    df_km_w.to_csv(csv_file_km_w, float_format='%.6f', encoding='utf-8')

    """ km """
    final_matrix_dist_m_w = apply_inverse_and_direction(raw_distance_matrix_km, dtype_conversion=1)
    df_m_w = pd.DataFrame(final_matrix_dist_m_w, index=index_labels, columns=index_labels)
    csv_file_m_w = path.join(base_location, 'dataset','section_distance_along_the_way_km.csv')
    df_m_w.to_csv(csv_file_m_w, float_format='%.4f', encoding='utf-8')

    # ====================================================================
    # II. 欧几里得距离矩阵 (D_euclidean) d = √((x₂ - x₁)² + (y₂ - y₁)² )
    # ====================================================================
    """ 坐标两点距离公式 【以下一个是km单位的版本，一个是米为单位的版本】"""
    index_labels = [f"P{i + 1} ({name})" for i, (name, x, y) in enumerate(section_point_data)]
    x_coords = np.array([x for name, x, y in section_point_data])
    y_coords = np.array([y for name, x, y in section_point_data])
    dx_sq = (x_coords[:, np.newaxis] - x_coords) ** 2
    dy_sq = (y_coords[:, np.newaxis] - y_coords) ** 2
    """ km """
    raw_euclidean_distance_matrix_m = np.sqrt(dx_sq + dy_sq)
    final_matrix_euclidean_m = apply_inverse_and_direction(raw_euclidean_distance_matrix_m, dtype_conversion=1/1000)
    df_euclidean_m = pd.DataFrame(final_matrix_euclidean_m, index=index_labels, columns=index_labels)
    csv_file_euclidean_m = os.path.join(base_location, 'dataset','euclidean_distance_km.csv')
    df_euclidean_m.to_csv(csv_file_euclidean_m, float_format='%.4f', encoding='utf-8')

    """ m """
    final_matrix_euclidean_km = apply_inverse_and_direction(raw_euclidean_distance_matrix_m, dtype_conversion=1)
    df_euclidean_km = pd.DataFrame(final_matrix_euclidean_km, index=index_labels, columns=index_labels)
    csv_file_euclidean_km = os.path.join(base_location, 'dataset', 'euclidean_distance_m.csv')
    df_euclidean_km.to_csv(csv_file_euclidean_km, float_format='%.6f', encoding='utf-8')


def apply_inverse_and_direction(raw_matrix, dtype_conversion=1.0):
    matrix = raw_matrix * dtype_conversion
    n = matrix.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse_matrix = np.nan_to_num(1.0 / matrix, posinf=0.0, neginf=0.0, nan=0.0)
    result_matrix = np.zeros_like(matrix)
    triu_indices = np.triu_indices(n, k=1)
    result_matrix[triu_indices] = inverse_matrix[triu_indices]
    tril_indices_mirror = (triu_indices[1], triu_indices[0])
    result_matrix[tril_indices_mirror] = -inverse_matrix[tril_indices_mirror]
    np.fill_diagonal(result_matrix, 15.0)
    return result_matrix

def process_csv_files(root_dir, null_dir_name, null_values):
    null_dir_path = os.path.join(root_dir, null_dir_name)
    if not os.path.exists(null_dir_path):
        os.makedirs(null_dir_path)
        print(f"已创建目标文件夹: {null_dir_path}")
    for filename in os.listdir(root_dir):
        if filename.endswith(".csv") and os.path.isfile(os.path.join(root_dir, filename)):
            file_path = os.path.join(root_dir, filename)
            if file_path.startswith(null_dir_path):
                continue
            print(f"\n--- 正在检查文件: {filename} ---")
            df = pd.read_csv(
                file_path,
                na_values=null_values,
                keep_default_na=True,
                encoding='utf-8'  # 尝试常见的编码，如果你的文件有编码问题可能需要更改
            )
            has_null = df.isnull().any(axis=None)
            if has_null:
                new_file_path = os.path.join(null_dir_path, filename)
                shutil.move(file_path, new_file_path)
                print(f"  **发现缺失值**，已移动到: {new_file_path}")
            else:
                print("  未发现缺失值。")

if __name__ == '__main__':
    NULL_VALUES = ['<null>', 'NULL', 'null', '', '#N/A', 'NA', 'NaN']
    TARGET_DIR = r"C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\do_nothing-8064-11199"
    NULL_DIR_NAME = "exist_null"
    process_csv_files(TARGET_DIR, NULL_DIR_NAME, NULL_VALUES)
    # section_matrix(r'C:\Users\Administrator\Desktop\mike-result-handle\assets')
    # init()
    # batch_gen_dataset(
    #     0,
    #     4031,
    #     r'C:\Users\Administrator\Desktop\mike-simulation-result-set\pump-0-4031',
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\pump-0-4031'
    # )
    # clip(
    #     0,
    #     4031,
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\pump-0-4031'
    # )


    # batch_gen_dataset(
    #     8064,
    #     11199,
    #     r'C:\Users\Administrator\Desktop\mike-simulation-result-set\do_nothing-8064-11199',
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\do_nothing-8064-11199'
    # )
    # clip(
    #     8064,
    #     11199,
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\do_nothing-8064-11199'
    # )

    # batch_gen_dataset(
    #     11200,
    #     15199,
    #     r'C:\Users\Administrator\Desktop\mike-simulation-result-set\gen-11200-15199',
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\gen-11200-15199'
    # )
    # clip(
    #     11200,
    #     15199,
    #     r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one\gen-11200-15199'
    # )



