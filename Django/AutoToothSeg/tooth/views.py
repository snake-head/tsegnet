from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import os
import sys
import numpy as np
from django.conf import settings
import json
from .utils.centroids import pred_centroids, pred_seg, pred_centroids2, true_centroids

sys.path.append(os.path.join(settings.BASE_DIR, 'tooth', 'models'))


# Create your views here.
def toothlist(request):
    upper_path = os.path.join(settings.LOCAL_FILE_DIR, 'upper')
    lower_path = os.path.join(settings.LOCAL_FILE_DIR, 'lower')
    upper_tooth_list = os.listdir(upper_path)
    lower_tooth_list = os.listdir(lower_path)
    for index, item in enumerate(upper_tooth_list):
        if not item.endswith('.stl'):
            upper_tooth_list.pop(index)
    for index, item in enumerate(lower_tooth_list):
        if not item.endswith('.stl'):
            lower_tooth_list.pop(index)
    tooth_list = dict()
    tooth_list["upper"] = upper_tooth_list
    tooth_list["lower"] = lower_tooth_list
    json_tooth_list = json.dumps(tooth_list, ensure_ascii=False)

    return JsonResponse({
        "status_code": 0,
        "data": json_tooth_list
    })


def import_upper_file(request, filename):
    if request.method == "GET":
        upper_path = os.path.join(settings.LOCAL_FILE_DIR, 'upper')
        file_path = os.path.join(upper_path, filename)
        with open(file_path, 'rb') as f:
            # print(f.read())
            return HttpResponse(f.read())

    if request.method == "POST":
        upper_path = os.path.join(settings.LOCAL_FILE_DIR, 'upper')
        export_path = os.path.join(settings.LOCAL_FILE_DIR, 'result')
        file_path = os.path.join(upper_path, filename)
        operate = request.POST.get("operate")
        if operate == 'predCentroids':
            centroids_pred, points, normal = pred_centroids(file_path)
            pred_data = dict()
            pred_data['points'] = points.tolist()
            pred_data['normal'] = normal.tolist()
            pred_data['centroidsPred'] = centroids_pred.tolist()
            json_pred_data = json.dumps(pred_data, ensure_ascii=False)
            return JsonResponse({
                "status_code": 0,
                "data": json_pred_data
            })
        elif operate == 'predCentroids2':
            centroids_pred, points, normal = pred_centroids2(file_path)
            pred_data = dict()
            pred_data['points'] = points.tolist()
            pred_data['normal'] = normal.tolist()
            print(centroids_pred)
            pred_data['centroidsPred'] = centroids_pred.tolist()
            json_pred_data = json.dumps(pred_data, ensure_ascii=False)
            return JsonResponse({
                "status_code": 0,
                "data": json_pred_data
            })
        elif operate == 'trueCentroids':
            centroids_pred, points, normal = true_centroids(upper_path, filename)
            pred_data = dict()
            pred_data['points'] = points.tolist()
            pred_data['normal'] = normal.tolist()
            print(centroids_pred)
            pred_data['centroidsPred'] = centroids_pred.tolist()
            json_pred_data = json.dumps(pred_data, ensure_ascii=False)
            return JsonResponse({
                "status_code": 0,
                "data": json_pred_data
            })
        elif operate == 'predSeg':
            points = (np.array((request.POST.get("points")).split(',')).astype(np.float32)).reshape(-1, 3)
            normal = (np.array((request.POST.get("normal")).split(',')).astype(np.float32)).reshape(-1, 3)
            centroids_pred = (np.array((request.POST.get("centroidsPred")).split(',')).astype(np.float32)).reshape(-1, 3)
            scene_data_plot, model_color = pred_seg(points, normal, centroids_pred, file_path, export_path, 'upper')
            pred_data = dict()
            pred_data['sceneData'] = scene_data_plot.tolist()
            pred_data['modelColor'] = model_color.tolist()
            json_pred_data = json.dumps(pred_data, ensure_ascii=False)
            return JsonResponse({
                "status_code": 0,
                "data": json_pred_data
            })


def import_lower_file(request, filename):
    lower_path = os.path.join(settings.LOCAL_FILE_DIR, 'lower')
    file_path = os.path.join(lower_path, filename)
    with open(file_path, 'rb') as f:
        # print(f.read())
        return HttpResponse(f.read())


def display_upper_result(request, filename, tooth):

    if request.method == "GET":
        result = os.path.join(settings.LOCAL_FILE_DIR, 'result', 'upper', filename, tooth)
        with open(result, 'rb') as f:
            # print(f.read())
            return HttpResponse(f.read())