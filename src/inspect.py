import json
import os

cases_location = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\cases.json'
cases_dict = dict()
REQUIRED_FILES = [
    "lhk.dfs0",
    "lhkhx.dfs0",
    "ygyj.dfs0",
    "z.dfs0",
    "LHKHX._spi_",
    "LHKHX.log",
    "LHKHX.m21fm",
    "LHKHX.mesh",
    "Manning.dfsu",
    "Qcs_LHKHX.dfs0",
    "Qout_LHK.dfs0",
    "Qout_YGYJ.dfs0"
]
REQUIRED_FILES_RESULT = [
    'lhkhx.dfs0',
    'lhk.dfs0',
    'ygyj.dfs0',
    'z.dfs0'
]
def init():
    cases = json.load(open(cases_location, 'r', encoding='utf-8'))
    cases_list = cases["cases"]
    global cases_dict
    cases_dict = {item['case_id']: item for item in cases_list}

def inspect(base_location : str,
            case_id : int):
    case_path = cases_dict[case_id]['path']
    case_type = cases_dict[case_id]['type']
    full_path = os.path.join(base_location, case_path)
    full_path_result = os.path.join(base_location, case_path, 'LHKHX.m21fm - Result Files')
    if not os.path.isdir(full_path):
        print(f'⚠{case_type}工况【{case_id}】模拟可能失败或未模拟需要重试：{case_path}')
        return False,
    if not os.path.isdir(full_path_result):
        print(f'⚠{case_type}工况【{case_id}】模拟可能失败,未见【LHKHX.m21fm - Result Files】文件夹')
        return False

    missing_files = []
    for filename in REQUIRED_FILES:
        file_path = os.path.join(full_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)

    missing_file_results = []
    for filename in REQUIRED_FILES_RESULT:
        file_path_ = os.path.join(full_path_result, filename)
        if not os.path.exists(file_path_):
            missing_file_results.append(filename)

    if missing_files:
        print(f'⚠{case_type}工况【{case_id}】模拟文件缺失：{case_path}')
        return False

    if missing_file_results:
        print(f'⚠{case_type}工况【{case_id}】模拟结果集文件缺失：{case_path}/LHKHX.m21fm - Result Files')
        return False

    log_filename = "LHKHX.log"
    log_file_path = os.path.join(full_path, log_filename)
    expected_end_line = "Normal run completion"

    lines = open(log_file_path, 'r', encoding='utf-8').readlines()
    if not lines:
        print(f'⚠{case_type}工况【{case_id}】MIKE日志文件为空，异常结束需要重新模拟：{case_path}')
        return False

    last_line = lines[-1].strip()

    if last_line == expected_end_line:
        return True
    else:
        print(f'⚠{case_type}工况【{case_id}】MIKE日志文件显示未正常结束：{last_line}')
        return False

def batch_inspect(start_id, end_id, base_location : str):
    init()
    for case_id in range(start_id, end_id + 1):
        inspect(base_location, case_id)


def main():
    pass

if __name__ == '__main__':
    main()