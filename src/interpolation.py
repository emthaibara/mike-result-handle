import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial import KDTree
from tqdm import tqdm


all_point_elevation = []
section_points = []
section_point_name_map = {}

""" 读取mesh文件、以及断面点信息 """
def r_data():
    mesh_file_path = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\LHKHX.mesh'
    section_file_path = r'C:\Users\Administrator\Desktop\mike-result-handle\assets\sections-info.csv'
    global all_point_elevation
    start_row = 2
    end_row = 26628
    line_count = 0
    mesh_file = open(mesh_file_path, 'r')
    for line in tqdm(mesh_file, desc='read mesh file'):
        line_count += 1
        if line_count < start_row or line_count > end_row:
            continue
        parts = line.split()
        if len(parts) >= 4:
            x = float(parts[1])
            y = float(parts[2])
            elevation = float(parts[3])
            all_point_elevation.append((x, y, elevation))

    section_df = pd.read_csv(section_file_path)
    for row in section_df.itertuples(index=False):
        name, x, y, d = row[0], row[1], row[2], row[3]
        section_points.append((x, y))
        section_point_name_map[(x,y)]=name

    return section_point_name_map

""" 空间插值：由于存在空间上的关系，我们使用最邻近的点来估算高程 """
def estimate_elevation_nearest_neighbor(all_points_data, profile_points):
    print(f"原始数据点数量: {all_points_data.shape[0]}")
    print(f"断面点数量: {profile_points.shape[0]}")

    source_coords = all_points_data[:, :2]
    source_elevations = all_points_data[:, 2]

    print("正在构建 KDTree...")
    kdtree = KDTree(source_coords)
    print("KDTree 构建完成。")

    print("正在搜索最近邻点...")
    distances, indices = kdtree.query(profile_points, k=1)
    print("搜索完成。")

    estimated = source_elevations[indices]
    elevation_map = {}
    __section_points = []
    for i in range(profile_points.shape[0]):
        x = profile_points[i, 0]
        y = profile_points[i, 1]
        z = estimated[i]
        elevation_map[(x, y)] = z
        __section_points.append((x, y, z))
    return elevation_map, __section_points, distances

if __name__ == '__main__':
    r_data()
    __all_points_data = np.array(all_point_elevation)
    __profile_points = np.array(section_points)
    estimated_elevations, section_points, __distances = estimate_elevation_nearest_neighbor(__all_points_data, __profile_points)

    profile_coords = np.array(list(estimated_elevations.keys()))
    profile_z = np.array(list(estimated_elevations.values()))

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    source_z = __all_points_data[:, 2]

    min_z = min(source_z.min(), profile_z.min())
    max_z = max(source_z.max(), profile_z.max())
    v_min = min_z - (max_z - min_z) * 0.05
    v_max = max_z + (max_z - min_z) * 0.05

    cmap_name = 'jet'
    cmap = get_cmap(cmap_name)

    # 距离参数
    min_dist = __distances.min()
    max_dist = __distances.max()
    base_size = 10
    scaling_factor = 500 / np.power(max_dist, 2)

    marker_sizes = base_size + scaling_factor * np.power(__distances, 2)

    # --- 3. 绘制断面点高程图 (叠加距离信息) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # 绘制断面点
    sc = ax.scatter(profile_coords[:, 0], profile_coords[:, 1],
                    c=profile_z,  # 颜色：表示估算的高程 Z
                    cmap=cmap,
                    vmin=v_min, vmax=v_max,
                    s=marker_sizes,  # **大小：直接表示最近邻距离**
                    marker='o',
                    edgecolors='red',
                    linewidths=1.5,
                    alpha=0.8)

    ax.set_title('断面点的估算高程与最近邻点距离分布图', fontsize=16)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, label='高程')

    # 图例句柄
    l_max = ax.scatter([], [], s=base_size + scaling_factor * np.power(max_dist, 2),
                       c='gray', edgecolors='red', label=f'最大距离 ≈ {max_dist:.2f}', alpha=0.8)
    l_mid = ax.scatter([], [], s=base_size + scaling_factor * np.power((min_dist + max_dist) / 2, 2),
                       c='gray', edgecolors='red', label=f'中等距离 ≈ {(min_dist + max_dist) / 2:.2f}', alpha=0.8)
    l_min = ax.scatter([], [], s=base_size + scaling_factor * np.power(min_dist, 2),
                       c='gray', edgecolors='red', label=f'最小距离 ≈ {min_dist:.2f}', alpha=0.8)

    ax.legend(handles=[l_max, l_mid, l_min],
              title='最近邻距离(对应估算范围)',
              loc='upper left',
              scatterpoints=1)

    fig.tight_layout()
    fig.savefig('estimate.png', dpi=600)
    plt.show()

