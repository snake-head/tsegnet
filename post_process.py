from visualizer.helper_data_plot import Plot as Plot
import os
import numpy as np
from struct import unpack
from visualizer.helper_data_plot import Plot as Plot
import open3d as o3d
from sklearn.cluster import DBSCAN
import cv2
from scipy.spatial import ConvexHull
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def BinarySTL(fname):
    '''Reads a binary STL file '''
    fp = open(fname, 'rb')
    Header = fp.read(80)
    nn = fp.read(4)
    Numtri = unpack('i', nn)[0]
    # print 'Number of triangles in the STL file: ',nn
    record_dtype = np.dtype([
        ('normals', np.float32, (3,)),
        ('Vertex1', np.float32, (3,)),
        ('Vertex2', np.float32, (3,)),
        ('Vertex3', np.float32, (3,)),
        ('atttr', '<i2', (1,))
    ])
    data = np.fromfile(fp, dtype=record_dtype, count=Numtri)
    fp.close()

    Normals = data['normals']
    Vertex1 = data['Vertex1']
    Vertex2 = data['Vertex2']
    Vertex3 = data['Vertex3']
    atttr = data['atttr']

    p = np.append(Vertex1, Vertex2, axis=0)
    p = np.append(p, Vertex3, axis=0)  # list(v1)
    Points = np.array(list(set(tuple(p1) for p1 in p)))

    return Vertex1, Vertex2, Vertex3, Normals, atttr, Header  # Header, Points, Normals,


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, '14.stl')

v1u_ft, v2u_ft, v3u_ft, vu_normals, atttr, Header = BinarySTL(file_dir)
ft_normal = np.sum(vu_normals, 0)
ft_normal /= np.sqrt(np.power(ft_normal, 2).sum(0))
z_normal = np.array([0, 0, 1])
theta = np.arccos(np.dot(ft_normal, z_normal))
axis = np.cross(ft_normal, z_normal)
axis /= np.sqrt(np.power(axis, 2).sum(0))

# rvec = -theta * axis
# rot_mat, _ = cv2.Rodrigues(rvec)
rot_mat = rotate_mat(axis, theta)

angle = np.arccos(np.dot(vu_normals, ft_normal)) * 180 / np.pi
index = np.argwhere(angle < 20)[:, 0]

ft = (v1u_ft + v2u_ft + v3u_ft) / 3
y = np.zeros(ft.shape[0])

# start = ft_normal * -2
# end = ft_normal * 2
# line = np.zeros((100, 3))
# center = np.mean(ft, 0)
# for i in range(100):
#     line[i, :] = start + (end - start)/100*i
# line += center
# y_line = np.ones(100)
# ft1 = np.concatenate((ft, line), 0)
# y1 = np.concatenate((y, y_line), 0)
# Plot.draw_pc_semins(pc_xyz=ft1, pc_semins=y1)

# y[index] = 1
# Plot.draw_pc_semins(pc_xyz=ft, pc_semins=y)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(ft)
# res = pcd.remove_radius_outlier(nb_points=20, radius=0.5)
# pcd = res[0]
# o3d.visualization.draw_geometries([pcd], #点云列表
#                                       window_name="离群点剔除",
#                                       point_show_normal=False,
#                                       width=800,  # 窗口宽度
#                                       height=600)  # 窗口高度

ft_proj = ft - ft * ft_normal * ft_normal
ft_proj_trans = ft_proj.transpose(1, 0)
ft_trans = ft.transpose(1, 0)
ft_xy = np.matmul(rot_mat, ft_trans).transpose(1, 0)
rot_normal = np.matmul(rot_mat, ft_normal)

ft_z = ft_xy[:, 2]
ft_z_max = np.max(ft_z)
ft_z_min = np.min(ft_z)
ft_nor = (ft_z - ft_z_min) / (ft_z_max - ft_z_min)
ft_low_index = np.argwhere(ft_nor < 0.3)[:, 0]
y[ft_low_index] = 1
Plot.draw_pc_semins(pc_xyz=ft_xy, pc_semins=y)
ft_xy[:, 2] = 0
# ft = ft_xy

ft_xy_2dim = ft_xy[:, :2]
ft_xy_2dim_edge = ft_xy[index, :2]
edge = ft[index, :2]
hull = ConvexHull(edge)

# plt.plot(ft_xy_2dim_edge[:, 0], ft_xy_2dim_edge[:, 1], 'o')
# for simplex in hull.simplices:
#     plt.plot(ft_xy_2dim_edge[simplex, 0], ft_xy_2dim_edge[simplex, 1], 'k-')
# plt.show()

# poly = ft_xy_2dim_edge[hull.vertices]
# keep = np.zeros(ft.shape[0])
# for i in range(ft.shape[0]):
#     p = ft_xy_2dim[i]
#     if is_in_poly(p, poly):
#         keep[i] = 1
# Plot.draw_pc_semins(pc_xyz=ft, pc_semins=keep)

# keep_index = np.argwhere(keep == 1)[:, 0]
# v1u_single = v1u_ft[keep_index]
# v2u_single = v2u_ft[keep_index]
# v3u_single = v3u_ft[keep_index]
# normal_single = vu_normals[keep_index]
# atttr_single = atttr[keep_index]
# Numtri = np.int32(v1u_single.shape[0])
# from struct import pack
# Numtri_str = pack('i', Numtri)
# str_stl = Header + Numtri_str
# for j in range(Numtri):
#     str_temp = pack('<12fh', normal_single[j, 0], normal_single[j, 1], normal_single[j, 2]
#                     , v1u_single[j, 0], v1u_single[j, 1], v1u_single[j, 2]
#                     , v2u_single[j, 0], v2u_single[j, 1], v2u_single[j, 2]
#                     , v3u_single[j, 0], v3u_single[j, 1], v3u_single[j, 2]
#                     , atttr_single[j][0])
#     str_stl = str_stl + str_temp
# save_path = os.path.join(base_dir, 'new.stl')
# fh = open(save_path, 'wb')
# str_final = str_stl.decode('utf8', 'ignore')
# fh.write(str_stl)
# fh.close()

# plt.plot(edge[:, 0], edge[:, 1], 'o')
# for simplex in hull.simplices:
#     plt.plot(edge[simplex, 0], edge[simplex, 1], 'k-')
# plt.show()
# y_edge = np.zeros(edge.shape[0])

# Plot.draw_pc_semins(pc_xyz=edge, pc_semins=y_edge)

# y = DBSCAN(eps=0.05, min_samples=3).fit_predict(ft_xy)

# Plot.draw_pc_semins(pc_xyz=ft, pc_semins=y)
print(1)
