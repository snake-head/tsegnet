import numpy as np
import vtk
from struct import unpack
import torch
import os
import importlib
from django.conf import settings
import glob

BATCH_SIZE = 16
NUM_POINT = 2048
NUM_CENTROIDS = 14
NUM_CLASSES = 16
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
experiment_dir = os.path.join(settings.LOG_DIR, '2022-05-04_23-48')
# experiment_dir = os.path.join(settings.LOG_DIR, '2022-04-28_12-44')
experiment_dir2 = os.path.join(settings.LOG_DIR, '2022-03-15_17-25')
experiment_dir3 = os.path.join(settings.LOG_DIR, '2022-03-06_17-09')


def BinarySTL_w(fname):
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


def gather_points(xyz, index):
    """
    :param xyz: pointcloud data, [B, x, N]
    :param index: cropping proposals index, [B, num_centroids, number of samples]
    :return: proposals_xyz: cropping points [B, num_centroids, x, number of samples]
    """
    batch_size, num_centroids, npoint = index.shape
    dimensions = xyz.shape[1]
    proposals_xyz = torch.zeros(batch_size, num_centroids, dimensions, npoint)
    for i in range(num_centroids):
        idx = index[:, i, :].unsqueeze(1).repeat(1, dimensions, 1)
        proposals_xyz[:, i, :, :] = torch.gather(xyz, 2, idx)
    return proposals_xyz


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

    p = np.append(Vertex1, Vertex2, axis=0)
    p = np.append(p, Vertex3, axis=0)  # list(v1)
    Points = np.array(list(set(tuple(p1) for p1 in p)))

    return Vertex1, Vertex2, Vertex3, Normals  # Header, Points, Normals,


def draw_stl(xyz, color):
    point_size = 10
    color_list = np.unique(color)
    cat = len(color_list)
    color_dict = dict()
    for i in range(cat):
        color_dict[color_list[i]] = i
    color_map = np.zeros((cat, 3))
    for i in range(cat):
        color_map[i] = 255 * np.random.rand(1, 3)

    rgb = np.zeros((len(color), 3))
    for i in range(len(color)):
        rgb[i][:] = color_map[int(color_dict[color[i]])][:]

    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    for i in range(0, len(xyz)):
        p = xyz[i]
        id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        Colors.InsertNextTuple3(rgb[i][0], rgb[i][1], rgb[i][2])

    point = vtk.vtkPolyData()
    point.SetPoints(points)
    point.SetVerts(vertices)
    point.GetPointData().SetScalars(Colors)
    point.Modified()

    return point


def draw_centroids(centroids, rgb):
    source = vtk.vtkSphereSource()
    source.SetCenter(centroids[0], centroids[1], centroids[2])
    source.SetRadius(1)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(rgb[0], rgb[1], rgb[2])
    return actor


def data_loader(stl_file):
    v1u_ft, v2u_ft, v3u_ft, vu_normals = BinarySTL(stl_file)
    vu_ft_tmp = np.vstack((v1u_ft, v2u_ft, v3u_ft))
    vu_ft = np.array(list(set([tuple(t) for t in vu_ft_tmp])))

    ft = np.floor(vu_ft * 10000000).astype(int)
    ft_ori = np.floor(vu_ft_tmp * 10000000).astype(int)

    vu_normal_ft = np.zeros((len(vu_ft), 3))
    vu_normal_final = np.zeros((len(vu_ft), 3))
    for k, item1 in enumerate(ft):
        index_ft = np.argwhere(ft_ori[:, 0] == item1[0])[:, 0]
        for j in range(index_ft.shape[0]):
            if item1[1] == ft_ori[index_ft[j], 1]:
                if item1[2] == ft_ori[index_ft[j], 2]:
                    temp_index = np.mod(index_ft[j], len(vu_ft))
                    vu_normal_ft[k, :] = vu_normal_ft[k, :] + vu_normals[temp_index, :]

    square = np.zeros(len(vu_ft))
    square[:] = np.sqrt(
        np.power(vu_normal_ft[:, 0], 2) + np.power(vu_normal_ft[:, 1], 2) + np.power(vu_normal_ft[:, 2], 2))
    vu_normal_final[:, 0] = vu_normal_ft[:, 0] / square
    vu_normal_final[:, 1] = vu_normal_ft[:, 1] / square
    vu_normal_final[:, 2] = vu_normal_ft[:, 2] / square

    return vu_ft, vu_normal_final


def toothDataset(points, normal, block_points=2048):
    point_set_ini = np.concatenate((points, normal), 1)
    points = point_set_ini[:, :6]
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    point_idxs = np.arange(0, np.size(points, 0), 1)
    num_batch = int(np.ceil(point_idxs.size / block_points))
    point_size = int(num_batch * block_points)
    replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
    point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
    point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
    np.random.shuffle(point_idxs)
    data_batch = points[point_idxs, :]
    normlized_xyz = np.zeros((point_size, 3))
    normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
    normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
    normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
    bias = [(coord_max[0] + coord_min[0]) / 2.0, (coord_max[1] + coord_min[1]) / 2.0]
    data_batch[:, 0] = data_batch[:, 0] - bias[0]
    data_batch[:, 1] = data_batch[:, 1] - bias[1]
    data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
    data_batch = data_batch.reshape((-1, block_points, data_batch.shape[1]))
    index_room = point_idxs.reshape((-1, block_points))
    return data_batch, coord_max, points, index_room, bias


def pred_centroid_utils(points, normal):
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CENTROIDS).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    scene_data, coords_max, _, _, _ = toothDataset(points, normal, block_points=2048)
    num_blocks = scene_data.shape[0]

    s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
    s_batch_num = 1
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
    sbatch_centroids = np.zeros((s_batch_num, BATCH_SIZE, NUM_CENTROIDS, 3))

    for sbatch in range(s_batch_num):
        start_idx = sbatch * BATCH_SIZE
        end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
        real_batch_size = end_idx - start_idx
        batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]

        torch_data = torch.Tensor(batch_data)
        torch_data = torch_data.float().cuda()
        torch_data = torch_data.transpose(2, 1)

        centroids_pred, distance_pred, l3_xyz = classifier(torch_data)
        # cluster_centroids = list()
        cluster_centroids = np.zeros((NUM_CENTROIDS, 3))
        batch_distance = distance_pred[0, :].transpose(1, 0)
        batch_centroids = centroids_pred[0, :, :].transpose(1, 0)
        index = torch.nonzero(batch_distance < 20)
        batch_centroids = batch_centroids[index[:, 0]]
        from sklearn.cluster import DBSCAN
        y = DBSCAN(eps=1.5, min_samples=5).fit_predict(batch_centroids.cpu().detach().numpy())

        batch_centroids = batch_centroids[np.argwhere(y > -1)[:, 0]]
        y = y[np.argwhere(y > -1)[:, 0]]
        # y = y + 1
        for i in range(np.max(y) + 1):
            cluster_centroid = batch_centroids[np.argwhere(y == i)[:, 0]].mean(0)
            # cluster_centroids.append(cluster_centroid.cpu().detach().numpy())
            if i == 0:
                cluster_centroids[:, :] = cluster_centroid.repeat(NUM_CENTROIDS, 1).cpu().detach().numpy()
            elif i >= NUM_CENTROIDS:
                continue
            else:
                cluster_centroids[i, :] = cluster_centroid.cpu().detach().numpy()


        # centroids_pred_np = centroids_pred.transpose(1, 2).cpu().detach().numpy()  # B, NUM_CENTROIDS, 3
        # centroids_pred_np[:, :, 0] *= coords_max[0]
        # centroids_pred_np[:, :, 1] *= coords_max[1]
        # centroids_pred_np[:, :, 2] *= coords_max[2]
        sbatch_centroids[sbatch, :, :, :] = cluster_centroids[None].repeat(BATCH_SIZE, axis=0)

    scene_data_plot = scene_data[:, :, :3].reshape(-1, 3)
    model_color = np.zeros(np.shape(scene_data_plot)[0])
    sbatch_centroids_plot = sbatch_centroids.reshape(-1, 3)
    pred_color = np.ones(np.shape(sbatch_centroids_plot)[0])
    return scene_data_plot, model_color, sbatch_centroids_plot, pred_color


def true_centroids_utils(single_tooth_path):
    all_centroids = np.zeros([14, 3])
    for k, up_teeth_name in enumerate(glob.glob(os.path.join(single_tooth_path, 'U*.stl'))):
        v1u_single, v2u_single, v3u_single, _ = BinarySTL(up_teeth_name)
        vu_single_tmp = np.vstack((v1u_single, v2u_single, v3u_single)).astype('float32')
        vu_single = np.array(list(set([tuple(t) for t in vu_single_tmp])))
        if k == 0:
            all_centroids[:, :] = np.mean(vu_single, axis=0).reshape(1, 3).repeat(14, 0)  # 不足14个的按第一个质心补足
        all_centroids[k, :] = np.mean(vu_single, axis=0)
    return all_centroids


def pred_seg_tsegnet_utils(points, normal, centroids):
    model_name = os.listdir(experiment_dir2 + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CENTROIDS).cuda()
    checkpoint = torch.load(str(experiment_dir2) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    vote_label_pool = np.zeros((points.shape[0], NUM_CLASSES))
    scene_data, _, whole_scene_data, scene_point_index, bias = toothDataset(points, normal, block_points=2048)
    num_blocks = scene_data.shape[0]

    s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

    batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
    # sbatch_points = np.zeros((s_batch_num, BATCH_SIZE, int(NUM_POINT / 4 * 14), 3))
    # sbatch_seg = np.zeros((s_batch_num, BATCH_SIZE, int(NUM_POINT / 4 * 14)))

    for sbatch in range(s_batch_num):
        start_idx = sbatch * BATCH_SIZE
        end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
        real_batch_size = end_idx - start_idx
        batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
        batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]

        torch_data = torch.Tensor(batch_data)
        torch_data = torch_data.float().cuda()
        torch_data = torch_data.transpose(2, 1)

        torch.cuda.empty_cache()
        seg_pred, proposal_index = classifier(torch_data, centroids)

        points2 = torch_data[:, :3, :]
        seg_proposal = np.argmax(seg_pred.contiguous().cpu().data, 3)
        batch_point_index_tensor = torch.Tensor(batch_point_index).float().cuda().unsqueeze(1)
        index_proposal = gather_points(batch_point_index_tensor, proposal_index).permute(0, 1, 3, 2)
        index_vote = index_proposal.reshape(BATCH_SIZE * NUM_CENTROIDS, -1)
        seg_pred_vote = seg_proposal.reshape(BATCH_SIZE * NUM_CENTROIDS, -1)
        vote_label_pool = add_vote(vote_label_pool, index_vote, seg_pred_vote, None)
        points_proposal = gather_points(points2, proposal_index).permute(0, 1, 3, 2)

    model_color = np.argmax(vote_label_pool, 1)
    scene_data_plot = whole_scene_data[:, :3]
    # 纠正偏移量
    # scene_data_plot[:, 0] = scene_data_plot[:, 0] + bias[0]
    # scene_data_plot[:, 1] = scene_data_plot[:, 1] + bias[1]
    return scene_data_plot, model_color


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def pred_seg_pointnet2_utils(points, normal):
    model_name = os.listdir(experiment_dir3 + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir3) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    scene_data, _, whole_scene_data, scene_point_index, _ = toothDataset(points, normal, block_points=2048)
    vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
    for _ in range(3):
        num_blocks = scene_data.shape[0]
        s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

        batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))

        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
            batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]

            torch_data = torch.Tensor(batch_data)
            torch_data = torch_data.float().cuda()
            torch_data = torch_data.transpose(2, 1)
            seg_pred, _ = classifier(torch_data)
            batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
            vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                       batch_pred_label[0:real_batch_size, ...], None)

    pred_label = np.argmax(vote_label_pool, 1)
    points_plot = whole_scene_data[:, :3]
    label_plot = pred_label
    return points_plot, label_plot


def get_centroids(points, target):
    """
    :param origin_points: [npoint, 3]
    :param target: [npoint]
    :return: ground truth centroids coords list
    """
    npoint = target.size
    all_centroids = list()

    a = np.unique(target)
    for k in a.tolist():  # except gingiva
        if k > 0:
            label = k
            index = np.argwhere(target == label)[:, 0]
            single_tooth = points[index]
            centroid = single_tooth.mean(0)
            all_centroids.append(centroid)
        else:
            continue
    all_centroids = np.array(all_centroids)
    return all_centroids


def create_stl(path, pc_all, ins_pred_all, export_path, loc):
    # return
    save_path = ''
    ins_uni = np.unique(ins_pred_all)
    ins_shape = ins_uni.shape[0]

    v1u_ft, v2u_ft, v3u_ft, vu_normals, atttr, Header = BinarySTL_w(path)
    v1u_new = np.floor(v1u_ft * 1000000).astype(int)
    v2u_new = np.floor(v2u_ft * 1000000).astype(int)
    v3u_new = np.floor(v3u_ft * 1000000).astype(int)
    for i in range(ins_shape):
        index_single = np.argwhere(ins_pred_all[:] == ins_uni[i])[:, 0]
        single_shape = index_single.shape[0]
        index_stl1 = []
        single = pc_all[index_single, :]
        single_new = np.floor(single * 1000000).astype(int)

        for k, item1 in enumerate(single_new):
            index_temp = np.argwhere(v1u_new[:, 0] == item1[0])[:, 0]
            for j in range(index_temp.shape[0]):
                if item1[1] == v1u_new[index_temp[j], 1]:
                    if item1[2] == v1u_new[index_temp[j], 2]:
                        index_stl1.append(index_temp[j])

        for k, item1 in enumerate(single_new):
            index_temp = np.argwhere(v2u_new[:, 0] == item1[0])[:, 0]
            for j in range(index_temp.shape[0]):
                if item1[1] == v2u_new[index_temp[j], 1]:
                    if item1[2] == v2u_new[index_temp[j], 2]:
                        index_stl1.append(index_temp[j])

        for k, item1 in enumerate(single_new):
            index_temp = np.argwhere(v3u_new[:, 0] == item1[0])[:, 0]
            for j in range(index_temp.shape[0]):
                if item1[1] == v3u_new[index_temp[j], 1]:
                    if item1[2] == v3u_new[index_temp[j], 2]:
                        index_stl1.append(index_temp[j])

        index_stl = np.array(index_stl1)

        index_uni = np.unique(index_stl)[1:].astype(int)
        v1u_single = v1u_ft[index_uni]
        v2u_single = v2u_ft[index_uni]
        v3u_single = v3u_ft[index_uni]
        normal_single = vu_normals[index_uni]
        atttr_single = atttr[index_uni]
        Numtri = np.int32(v1u_single.shape[0])

        from struct import pack
        Numtri_str = pack('i', Numtri)
        # record_dtype = np.dtype([
        #     ('normals', np.float32, (3,)),
        #     ('Vertex1', np.float32, (3,)),
        #     ('Vertex2', np.float32, (3,)),
        #     ('Vertex3', np.float32, (3,)),
        #     ('atttr', '<i2', (1,))
        # ])
        str_stl = Header + Numtri_str
        for j in range(Numtri):
            str_temp = pack('<12fh', normal_single[j, 0], normal_single[j, 1], normal_single[j, 2]
                            , v1u_single[j, 0], v1u_single[j, 1], v1u_single[j, 2]
                            , v2u_single[j, 0], v2u_single[j, 1], v2u_single[j, 2]
                            , v3u_single[j, 0], v3u_single[j, 1], v3u_single[j, 2]
                            , atttr_single[j][0])
            str_stl = str_stl + str_temp
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        loc_path = os.path.join(export_path, loc)
        if not os.path.exists(loc_path):
            os.makedirs(loc_path)
        filename = path.split('\\')[-1].split('.')[0]
        save_path = os.path.join(loc_path, filename)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path1 = os.path.join(save_path, str(i) + '.stl')
        fh = open(path1, 'wb')
        str_final = str_stl.decode('utf8', 'ignore')
        fh.write(str_stl)
        fh.close()
