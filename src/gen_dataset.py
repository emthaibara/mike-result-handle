from os import path
import mikeio
import pandas as pd

# 定义文件所在的根目录
location = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\generated\pump-simulation\z0-1-[2605.0]\q1-6-[520.0]\q2-1-[-625.0]\q3-8-[-750.0]\LHKHX.m21fm - Result Files'

base_path = path.join(path.dirname(path.dirname(__file__)),'assets','generated')
dataset_base_path = path.join(path.dirname(path.dirname(__file__)),'dataset')
type_map = {
    '抽水':'pump-simulation',
    '发电':'gen-simulation',
    '不抽不发':'nothing-simulation'
}
def processed_result(case_id: int):
    lhk_location = path.join(base_path, type_map[cases[case_id]['type']], cases[case_id]['path'], 'LHKHX.m21fm - Result Files', 'lhk.dfs0')
    lhkhx_location = path.join(base_path, type_map[cases[case_id]['type']], cases[case_id]['path'], 'LHKHX.m21fm - Result Files', 'lhkhx.dfs0')
    ygyj_location = path.join(base_path, type_map[cases[case_id]['type']], cases[case_id]['path'], 'LHKHX.m21fm - Result Files', 'ygyj.dfs0')
    z_location = path.join(base_path, type_map[cases[case_id]['type']], cases[case_id]['path'], 'LHKHX.m21fm - Result Files', 'z.dfs0')





    # 1. 读取所有 dfs0 文件并转换为 pandas DataFrame
    print("开始加载 dfs0 文件...")
    df_lhk = mikeio.read(path.join(location, 'lhk.dfs0')).to_dataframe()
    df_lhk.columns = [f'{col}_lhk' for col in df_lhk.columns]

    df_lhkhx = mikeio.read(path.join(location, 'lhkhx.dfs0')).to_dataframe()
    df_lhkhx.columns = [f'{col}_lhkhx' for col in df_lhkhx.columns]

    df_ygyj = mikeio.read(path.join(location, 'ygyj.dfs0')).to_dataframe()
    df_ygyj.columns = [f'{col}_ygyj' for col in df_ygyj.columns]

    df_z = mikeio.read(path.join(location, 'z.dfs0')).to_dataframe()

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

    master_df = df_lhk.merge(df_lhkhx, left_index=True, right_index=True, how='outer')
    master_df = master_df.merge(df_ygyj, left_index=True, right_index=True, how='outer')
    master_df = master_df.merge(df_z, left_index=True, right_index=True, how='outer')

    print("文件加载和数据对齐完成。")
    output_df = pd.DataFrame(index=df_z.index, columns=cross_sections)
    # 遍历每一个时间点
    for time_index, row in master_df.iterrows():
        for section_name in cross_sections:
            # 使用重命名后的列名来获取数据
            val_lhk_1 = row[df_lhk.columns[0]]
            val_lhk_2 = row[df_lhk.columns[1]]
            val_lhkhx_1 = row[df_lhkhx.columns[0]]
            val_lhkhx_2 = row[df_lhkhx.columns[1]]
            val_ygyj_1 = row[df_ygyj.columns[0]]
            val_ygyj_2 = row[df_ygyj.columns[1]]
            # 获取 z.dfs0 对应截面的值，使用原始列名
            val_z = row[section_name]

            # 组合成元组并赋值给新 DataFrame
            output_df.loc[time_index, section_name] = (
                val_lhk_1, val_lhk_2,
                val_lhkhx_1, val_lhkhx_2,
                val_ygyj_1, val_ygyj_2,
                val_z
            )
    print("新 DataFrame 构建完成。")
    # 5. 导出为 CSV
    output_df.to_csv(path.join(location, f'case_{case_id}_result.csv'), index=True, index_label='time',
                     encoding='utf-8')
    output_df.to_excel(path.join(location, f'case_{case_id}_result.xlsx'), index=True, index_label='time', engine='openpyxl')
    print(f"结果已成功保存到 {path.join(location, f'case_{case_id}_result.csv')}")

cases = dict()
def load_cases():
    pass

processed_result(470)




