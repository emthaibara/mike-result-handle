import json
import os
import shutil
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
    output_columns = ['q1_origin', 'q2_origin', 'q3_origin', 'q1', 'q2', 'q3']

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
    output_df['q1_origin'] = q1_origin
    output_df['q2_origin'] = q2_origin
    output_df['q3_origin'] = q3_origin

    output_df['q1'] = master_df[df_lhk.columns[0]].abs().round(1)

    if q2_origin == 0:
        output_df['q2'] = 0.0
    else:
        q2_sim = master_df[df_lhkhx.columns[0]].abs().round(1)
        if q2_origin < 0:
            output_df['q2'] = -q2_sim
        elif q2_origin > 0:
            output_df['q2'] = q2_sim

    output_df['q3'] = -master_df[df_ygyj.columns[0]].abs().round(1)

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

def section_matrix():
    section_distance_data = [
        ('YG63LHK', 0.2), ('YG62-1.5', 0.5), ('YG62-1', 0.8), ('YG62-0.5', 1.0), ('YG62', 1.2),
        ('YG61-2.5', 1.3), ('YG61-2', 1.5), ('YG61-1.5', 1.6), ('YG61-1', 1.8), ('YG61-0.9HX', 1.9),
        ('YG61-0.75', 2.0), ('YG61-0.5', 2.1), ('YG61-0.25', 2.2), ('YG61', 2.3), ('YG60-1.75', 2.5),
        ('YG60-1.5', 2.7), ('YG60-1.25', 2.9), ('YG60-1', 3.0), ('YG60-0.5', 3.3), ('YG60', 3.5),
        ('YG59-0.5', 3.7), ('YG58-1', 3.9), ('YG59-2', 4.2), ('YG59', 4.6), ('YG58-1', 5.1),
        ('YG58', 5.5), ('YG57-1', 5.9), ('YG57-2', 6.4), ('YG57', 6.8), ('YG56-1', 7.3),
        ('YG56', 7.8), ('YG55-1', 8.2), ('YG55', 8.6), ('YG54-1', 9.2), ('YG54', 9.9),
        ('YG53-2', 10.3), ('YG53-1', 10.7), ('YG53', 11.0), ('YG52-1', 11.6), ('YG52', 12.3),
        ('YG51-1', 12.7), ('YG51', 13.2), ('YG50-1', 13.6), ('YG50', 14.0), ('YG49-1', 14.4),
        ('YG49', 14.9), ('YG48-1', 15.4), ('YG48', 16.0), ('YG47-2', 16.3), ('YG47', 16.8),
        ('YG47-1', 17.3), ('YG46', 17.8), ('YG46-1', 18.3), ('YG45-1', 18.9), ('YG45', 19.3),
        ('YG44YGY', 19.5)
    ]
    section_point_data = [
        ('YG63LHK', 269025.47, 3742385.49), ('YG62-1.5', 269061.84, 3742137.24), ('YG62-1', 269104.79, 3741865.62),
        ('YG62-0.5', 269023.87, 3741608.03), ('YG62', 269001.44, 3741467.83), ('YG61-2.5', 269049.20, 3741308.93),
        ('YG61-2', 269100.51, 3741190.69), ('YG61-1.5', 269217.13, 3741073.39), ('YG61-1', 269325.96, 3740986.03),
        ('YG61-0.9HX', 269409.75, 3741031.69), ('YG61-0.75', 269451.12, 3740853.87),
        ('YG61-0.5', 269506.00, 3740766.95),
        ('YG61-0.25', 269551.89, 3740623.33), ('YG61', 269556.03, 3740482.84), ('YG60-1.75', 269490.09, 3740318.31),
        ('YG60-1.5', 269409.51, 3740158.92), ('YG60-1.25', 269305.36, 3740041.46), ('YG60-1', 269209.09, 3739889.26),
        ('YG60-0.5', 269053.05, 3739685.16), ('YG60', 268919.82, 3739476.30), ('YG59-0.5', 268832.45, 3739996.60),
        ('YG58-1', 268746.06, 3739100.14), ('YG59-2', 268649.77, 3738843.33), ('YG59', 268506.24, 3738531.31),
        ('YG58-1', 268231.40, 3738138.69), ('YG58', 267851.96, 3737914.43), ('YG57-1', 267547.56, 3737716.21),
        ('YG57-2', 267688.57, 3737265.74), ('YG57', 267855.26, 3736816.18), ('YG56-1', 267602.80, 3736282.34),
        ('YG56', 267533.59, 3735809.92), ('YG55-1', 267445.63, 3735413.68), ('YG55', 267349.74, 3735037.58),
        ('YG54-1', 267411.35, 3734497.89), ('YG54', 267996.18, 3734128.87), ('YG53-2', 268834.62, 3734026.49),
        ('YG53-1', 268658.98, 3733866.58), ('YG53', 268888.28, 3733642.17), ('YG52-1', 269500.43, 3733578.18),
        ('YG52', 270124.84, 3733571.19), ('YG51-1', 270433.90, 3733772.34), ('YG51', 271028.25, 3733754.68),
        ('YG50-1', 271077.19, 3733391.61), ('YG50', 270886.89, 3733066.72), ('YG49-1', 270475.23, 3732872.68),
        ('YG49', 270035.92, 3732820.53), ('YG48-1', 269468.21, 3732757.47), ('YG48', 269091.97, 3732544.99),
        ('YG47-2', 269141.71, 3732133.28), ('YG47', 269034.05, 3731712.22), ('YG47-1', 268472.48, 3731712.10),
        ('YG46', 268011.37, 3731597.97), ('YG46-1', 268124.50, 3731123.31), ('YG45-1', 268491.42, 3730761.85),
        ('YG45', 268611.04, 3730335.80), ('YG44YGY', 268543.22, 3730146.31)
    ]

    base_location = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset'
    """ 
        两种方式计算断面点之间的间距：
                    1. 根据【据两河口坝址距离km】做差取绝对值，得到两个断面点之间的沿程距离
                    2. 根据【（X,Y）】两点间距离公式计算，得到两个断面点的直线距离
    """
    """ 两个断面点与两河口水坝的距离（沿程距离）做差取绝对值 【以下一个是km单位的版本，一个是米为单位的版本】"""
    index_labels = [f"P{i + 1} ({name})" for i, (name, d) in enumerate(section_distance_data)]
    distances_km = np.array([d for name, d in section_distance_data])
    distance_matrix_km = np.abs(distances_km[:, np.newaxis] - distances_km)
    df_km = pd.DataFrame(distance_matrix_km, index=index_labels, columns=index_labels)
    csv_file_km = path.join(base_location,'section_distance_along_the_way_km.csv')
    df_km.to_csv(csv_file_km, float_format='%.1f', encoding='utf-8')
    """ km """
    distance_matrix_m = distance_matrix_km * 1000
    df_m = pd.DataFrame(distance_matrix_m, index=index_labels, columns=index_labels)
    csv_file_m = path.join(base_location, 'section_distance_along_the_way_m.csv')
    df_m.to_csv(csv_file_m, float_format='%.2f', encoding='utf-8')

    """ 坐标两点距离公式 【以下一个是km单位的版本，一个是米为单位的版本】"""
    index_labels = [f"P{i + 1} ({name})" for i, (name, x, y) in enumerate(section_point_data)]
    x_coords = np.array([x for name, x, y in section_point_data])
    y_coords = np.array([y for name, x, y in section_point_data])
    dx_sq = (x_coords[:, np.newaxis] - x_coords) ** 2
    dy_sq = (y_coords[:, np.newaxis] - y_coords) ** 2
    """ m """
    euclidean_distance_matrix_m = np.sqrt(dx_sq + dy_sq)
    df_euclidean = pd.DataFrame(
        euclidean_distance_matrix_m,
        index=index_labels,
        columns=index_labels
    )
    csv_file_name = path.join(base_location, 'euclidean_distance_m.csv')
    df_euclidean.to_csv(csv_file_name, float_format='%.2f', encoding='utf-8')
    """ km """
    euclidean_distance_matrix_km = euclidean_distance_matrix_m / 1000
    df_km = pd.DataFrame(
        euclidean_distance_matrix_km,
        index=index_labels,
        columns=index_labels
    )
    csv_file_km = path.join(base_location, 'euclidean_distance_km.csv')
    df_km.to_csv(csv_file_km, float_format='%.1f', encoding='utf-8')

if __name__ == '__main__':
    # section_matrix()
    __base_location = r'C:\Users\Administrator\Desktop\mike-simulation-result-set\pump-0-4031'
    __dataset_location = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\dataset\batch-one'
    init()

    batch_gen_dataset(0,
                      4031,
                      __base_location,
                      __dataset_location)
    clip(0,
         4031,
         __dataset_location)

    __base_location = r'C:\Users\Administrator\Desktop\mike-simulation-result-set\do_nothing-8064-11199'
    batch_gen_dataset(8064,
                      11199,
                      __base_location,
                      __dataset_location)

    __base_location = r'C:\Users\Administrator\Desktop\mike-simulation-result-set\gen-11200-15199'
    batch_gen_dataset(11200,
                      15199,
                      __base_location,
                      __dataset_location)
    clip(11200,
         15199,
         __dataset_location)


