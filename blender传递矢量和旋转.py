import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as R
import bpy
# 安装库的指令：python -m pip install PyMuPDF Pillow scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 需要修改的配置参数
web = ["90t", "196.01h,85.56t", "119.75h,89.6t"]  # 平面图网址中的 h 值与 t 值
L = 8192  # 2:1 全景图像的短边像素值
node_group = bpy.data.node_groups.get("1")  # 几何节点树名称
mn_values = np.array([[3895, 4082], [12805, 4316], [9347, 4075]])  # 平面图白色中线交点位于全景图的像素坐标
H = 2  # 标准高度体的高度
ac_lines = [[12805, 4594], [12805, 3966]]  # 标准高度体的底部和顶部在全景图中的像素坐标

# 一般不需要修改的配置参数
C1 = np.array([0, 0, 0])  # 点 C1 的坐标
C2 = np.array([0, 0, H])  # 点 C2 的坐标
r = 1  # 单位向量的长度
initial_guess = np.eye(3).flatten()  # 初始旋转矩阵的猜测
optimization_method = 'Nelder-Mead'  # 优化方法

def parse_web_data(web):
    """解析平面图网址中的 h 值与 t 值"""
    rotation_vectors = []
    
    for entry in web:
        h_value = 0.0
        t_value = 0.0
        
        parts = entry.split(',')
        
        for part in parts:
            if 'h' in part:
                h_value = float(part.replace('h', ''))
            elif 't' in part:
                t_value = float(part.replace('t', ''))
        
        rotation_vectors.append((h_value, t_value))
    
    return rotation_vectors

def direction_vector(h, t):
    """计算方向向量"""
    h_rad, t_rad = np.deg2rad(h), np.deg2rad(t - 90)
    return np.array([np.cos(h_rad) * np.cos(t_rad), -np.sin(h_rad) * np.cos(t_rad), np.sin(t_rad)])

def calculate_direction(r, m, n, l):
    """计算方向向量的实际坐标"""
    pi = np.pi
    sin_n, cos_n = np.sin(n * pi / l), np.cos(n * pi / l)
    sin_m, cos_m = np.sin(m * pi / l), np.cos(m * pi / l)
    return np.array([-sin_n * cos_m * r, sin_n * sin_m * r, cos_n * r])

def residuals(rot_mat_flat, A, b):
    """计算残差函数"""
    rot_mat = rot_mat_flat.reshape(3, 3)
    return (rot_mat @ A - b).flatten()

rotation_vectors = parse_web_data(web)

# 计算旋转向量和方向向量
rot_dirs = np.hstack([direction_vector(h, t).reshape(-1, 1) for h, t in rotation_vectors])
dirs = np.hstack([calculate_direction(r, m, n, L).reshape(-1, 1) for m, n in mn_values])

# 构建矩阵 A 和向量 b
A, b = dirs, rot_dirs

# 优化旋转矩阵
opt_result = least_squares(residuals, initial_guess, args=(A, b))
recovered_rot_mat = opt_result.x.reshape(3, 3)

# 正交化旋转矩阵并调整行列式
U, _, Vt = np.linalg.svd(recovered_rot_mat)
orthogonal_rot_mat = U @ Vt
if np.linalg.det(orthogonal_rot_mat) < 0:
    Vt[-1] *= -1
    orthogonal_rot_mat = U @ Vt

# 转换为四元数
rotation = R.from_matrix(orthogonal_rot_mat)
quaternion = rotation.as_quat()  # [x, y, z, w] 格式
quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

# 转换为欧拉角
euler_angles = rotation.as_euler('xyz', degrees=True)

# 计算坐标
ac1, ac2 = [orthogonal_rot_mat @ calculate_direction(1, m, n, L) for m, n in ac_lines]
C1, C2 = C1, C2

def distance_between_lines(params, ac1, ac2, C1, C2):
    """计算两条线之间的距离"""
    t1, t2 = params
    point1, point2 = C1 + t1 * ac1, C2 + t2 * ac2
    return np.linalg.norm(point1 - point2)

# 最小化两条线之间的距离
result = minimize(distance_between_lines, [0, 0], args=(ac1, ac2, C1, C2), method=optimization_method)
t1_opt, t2_opt = result.x
A1, A2 = C1 + t1_opt * ac1, C2 + t2_opt * ac2
A_avg = (A1 + A2) / 2

# 打印结果
print("恢复的旋转矩阵 R:", orthogonal_rot_mat)
print("四元数 (w, x, y, z):", quaternion_wxyz)
print("欧拉角 (x, y, z) in degrees:", euler_angles)
print("点 A 坐标:", A_avg)

# Blender 脚本部分
# 获取并设置矢量节点的属性
vector_node = node_group.nodes.get("矢量")
if vector_node:
    vector_node.vector = tuple(A_avg)  # 将 A_avg 的 xyz 值传递给矢量节点

# 获取并设置旋转节点的属性
rotation_node = node_group.nodes.get("旋转")
if rotation_node:
    # 将欧拉角转换为弧度
    euler_radians = np.deg2rad(euler_angles)
    rotation_node.rotation_euler = tuple(euler_radians)  # 将转换后的欧拉角传递给旋转节点
