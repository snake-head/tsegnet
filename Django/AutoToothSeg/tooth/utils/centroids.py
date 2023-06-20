from .seg_utils import data_loader, pred_centroid_utils, draw_stl, draw_centroids, pred_seg_tsegnet_utils \
    , pred_seg_pointnet2_utils, create_stl, pred_seg_pointnet2_utils, get_centroids, true_centroids_utils
import os


def pred_centroids(path):
    points, normal = data_loader(path)
    scene_data_plot, model_color, centroids_pred, pred_color = pred_centroid_utils(points, normal)
    # scene_data_plot (n*3) model_color (n)
    # point = draw_stl(scene_data_plot, model_color) # 生成polydata（python vtk对象）
    return centroids_pred, points, normal


def pred_centroids2(path):
    points, normal = data_loader(path)
    scene_data_plot, model_color = pred_seg_pointnet2_utils(points, normal)
    centroids_pred = get_centroids(scene_data_plot, model_color)
    return centroids_pred, points, normal


def true_centroids(path, filename):
    whole_model = os.path.join(path, filename)
    single_tooth = os.path.join(path, filename.split('.')[0])
    points, normal = data_loader(whole_model)
    centroids_true = true_centroids_utils(single_tooth)
    return centroids_true, points, normal


def pred_seg(points, normal, centroids_pred, path, export_path, loc):
    scene_data_plot, model_color = pred_seg_tsegnet_utils(points, normal, centroids_pred)
    create_stl(path, scene_data_plot, model_color, export_path, loc)

    return scene_data_plot, model_color
