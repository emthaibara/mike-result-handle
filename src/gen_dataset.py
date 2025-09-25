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

    """ qx_origin代表原始qx流量 """
    q1_origin = cases_dict[case_id]['q1-flow_rate']
    q2_origin = cases_dict[case_id]['q2-flow_rate']
    q3_origin = cases_dict[case_id]['q3-flow_rate']
    output_df['q1_origin'] = q1_origin
    output_df['q2_origin'] = q2_origin
    output_df['q3_origin'] = q3_origin

    output_df['q1'] = master_df[df_lhk.columns[0]].round(1)
    if q2_origin == 0 :
        output_df['q2'] = 0
    elif q2_origin < 0 :
        output_df['q2'] = -master_df[df_lhkhx.columns[0]].round(1)
    elif q2_origin > 0 :
        output_df['q2'] = master_df[df_lhkhx.columns[0]].round(1)

    if q3_origin < 0 :
        output_df['q3'] = -master_df[df_ygyj.columns[0]].round(1)
    elif q3_origin > 0 :
        output_df['q3'] = master_df[df_ygyj.columns[0]].round(1)

    for section_name in cross_sections:
        output_df[section_name] = master_df[section_name]
    output_df.to_csv(path.join(dataset_location, f'{case_id}.csv'), index=True, index_label='time', encoding='utf-8')

def move_only_csv_files(source_folder, destination_folder):
    if not os.path.isdir(source_folder):
        print(f"错误: 源文件夹 '{source_folder}' 不存在或不是一个目录。")
        return False
    if not os.path.exists(destination_folder):
        try:
            os.makedirs(destination_folder)
            print(f"创建目标文件夹: '{destination_folder}'")
        except OSError as e:
            print(f"错误: 无法创建目标文件夹 '{destination_folder}' - {e}")
            return False
    print(f"开始移动 '{source_folder}' 中的 .csv 文件到 '{destination_folder}'...")
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        # 核心逻辑：判断是否是文件且文件扩展名是否为 .csv
        if os.path.isfile(source_path) and item.endswith('.csv'):
            try:
                shutil.move(source_path, destination_path)
                print(f"已移动 .csv 文件: {item}")
            except Exception as e:
                print(f"移动文件 '{item}' 时发生错误: {e}")
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

cases_location = r'C:\Users\lemt\PycharmProjects\mike-result-handle\assets\cases.json'
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

if __name__ == '__main__':
    __base_location = r'E:\mike-simulation-result-set\pump-0-4031'
    __dataset_location = r'C:\Users\lemt\PycharmProjects\mike-result-handle\assets\dataset\batch-one\pump-0-4031'
    init()
    batch_gen_dataset(0,
                      4031,
                      __base_location,
                      __dataset_location)
    clip(0,
         4031,
         __dataset_location)

    # __base_location = r'Y:\mike-simulation-result-set\do_nothing-8064-11199'
    # __dataset_location = r'C:\Users\lemt\PycharmProjects\mike-result-handle\assets\dataset\batch-one\do_nothing-8064-11199'
    # batch_gen_dataset(8064,
    #                   11199,
    #                   __base_location,
    #                   __dataset_location)
    #
    # __base_location = r'Y:\mike-simulation-result-set\gen-11200-15199'
    # __dataset_location = r'C:\Users\lemt\PycharmProjects\mike-result-handle\assets\dataset\batch-one\gen-11200-15199'
    # batch_gen_dataset(11200,
    #                   15199,
    #                   __base_location,
    #                   __dataset_location)
    # clip(11200,
    #      15199,
    #      __dataset_location)


