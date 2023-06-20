import torch.nn as nn
import torch.nn.functional as F
from tsegnet_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, get_centroids
import torch
import numpy as np


class get_model(nn.Module):
    def __init__(self, num_centroids):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [2.5, 5], [16, 32], 9 + 3, [[9, 32, 32], [9, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [5, 10], [16, 32], 64 + 3, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [10, 20], [16, 32], 256 + 3, [[256, 196, 256], [256, 196, 256]])

        self.displacement = nn.Sequential(
            nn.Conv1d(515, 256, (1,)),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 3, (1,)),
            nn.BatchNorm1d(3),
            nn.ReLU(True),
        )

        self.distance = nn.Sequential(
            nn.Conv1d(515, 256, (1,)),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 1, (1,)),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
        )

        # self.centroid = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(64, num_centroids * 3),
        #     nn.Conv1d(4, 3, (243,))
        # )

        self.num_centroids = num_centroids

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # l0_xyz_np = l0_xyz.transpose(1, 2).cpu().detach().numpy()
        # for i in range(16):
        #     l0_xyz_plot = l0_xyz_np[i].reshape(-1, 3)
        #     l0_color = np.zeros(np.shape(l0_xyz_plot)[0]) * 3
        #     Plot.draw_pc_semins(pc_xyz=l0_xyz_plot[:, :], pc_semins=l0_color)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        # l1_xyz_np = l1_xyz.transpose(1, 2).cpu().detach().numpy()
        # for i in range(16):
        #     l1_xyz_plot = l1_xyz_np[i].reshape(-1, 3)
        #     l1_color = np.zeros(np.shape(l1_xyz_plot)[0]) * 3
        #     Plot.draw_pc_semins(pc_xyz=l1_xyz_plot[:, :], pc_semins=l1_color)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l3_xyz_np = l3_xyz.transpose(1, 2).cpu().detach().numpy()
        # for i in range(16):
        #     l3_xyz_plot = l3_xyz_np[i].reshape(-1, 3)
        #     l3_color = np.zeros(np.shape(l3_xyz_plot)[0]) * 3
        #     Plot.draw_pc_semins(pc_xyz=l3_xyz_plot[:, :], pc_semins=l3_color)

        l3 = torch.cat((l3_xyz, l3_points), 1)
        displacement = self.displacement(l3)    # 16,3,256
        distance = self.distance(l3)    # 16,1,256
        centroids = l3_xyz + displacement
        # batch_num = xyz.shape[0]
        # cluster_centroids = torch.empty(batch_num, self.num_centroids, 3)
        # for b in range(batch_num):
        #     batch_distance = distance[b, :].transpose(1, 0)
        #     batch_centroids = centroids[b, :, :].transpose(1, 0)
        #     index = torch.nonzero(batch_distance < 20)
        #     batch_centroids = batch_centroids[index[:, 0]]
        #     y = DBSCAN(eps=1.5, min_samples=5).fit_predict(batch_centroids.cpu().detach().numpy())
        #     Plot.draw_pc_semins(pc_xyz=batch_centroids.cpu().detach().numpy(), pc_semins=y)
        #     # y = y[np.argwhere(y > -1)[:, 0]]
        #     y = y + 1
        #     for i in range(np.max(y) + 1):
        #         cluster_centroid = batch_centroids[np.argwhere(y == i)[:, 0]].mean(0)
        #         if i == 0:
        #             cluster_centroids[b, :, :] = cluster_centroid.repeat(self.num_centroids, 1)
        #         elif i >= self.num_centroids:
        #             continue
        #         else:
        #             cluster_centroids[b, i, :] = cluster_centroid
        #
        # centroids_pred = cluster_centroids.cuda().transpose(2, 1)
            # displacement = torch.cat((l3_xyz, displacement), 1)
        # displacement = displacement.view(displacement.size(0), -1)
        # centroids = self.centroid(displacement)
        # centroids = centroids.view(centroids.size(0), 3, self.num_centroids)
        return centroids, distance, l3_xyz


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, displacement_pred, subsampled_points, centroids_pred, origin_points, target):
        """
        displacement_pred:  [B, num of subsampled points]
        subsampled_points:  [B, 3, num of subsampled points]
        centroids_pred: [B, 3, 14]
        origin_points: [B, 9, npoint=4096]
        target: [B, npoint=4096]
        """
        origin_points = origin_points.transpose(1, 2)[:, :, :3]    # 使用归一化的坐标
        centroids_gt = get_centroids(origin_points, target)
        num_centroids_pred = centroids_pred.size(2)
        num_points = displacement_pred.size(1)
        batch_num = displacement_pred.size(0)
        total_loss = 0

        # centroids_pred_np = centroids_pred.transpose(1, 2).cpu().detach().numpy()
        # batch_data = origin_points[:, :3, :].transpose(1, 2).cpu().detach().numpy()
        # sbatch_centroids_plot = centroids_pred_np[0, :, :]
        # pred_color = np.ones(np.shape(sbatch_centroids_plot)[0])
        # scene_data_plot = batch_data[0, :, :3].reshape(-1, 3)
        # model_color = np.ones(np.shape(scene_data_plot)[0]) * 2
        # pc_all = np.concatenate((sbatch_centroids_plot, scene_data_plot), axis=0)
        # label = np.concatenate((pred_color, model_color), axis=0)
        # Plot.draw_pc_semins(pc_xyz=pc_all[:, :], pc_semins=label)

        def smooth(x):
            x_abs = x.abs()
            smooth1 = torch.where(x_abs < 1, 0.5 * torch.pow(x, 2), torch.abs(x) - 0.5)
            return smooth1

        for i in range(batch_num):
            single_centroids = (torch.stack(centroids_gt[i])).float().cuda().transpose(0, 1)  # 3, num_centroids
            num_centroids = single_centroids.size(1)
            """distance estimate loss"""
            subsample_pos_ex = subsampled_points[i, :, :].unsqueeze(2).repeat(1, 1, num_centroids)
            target_pos_ex = single_centroids.unsqueeze(1).repeat(1, num_points, 1)
            distance, _ = (subsample_pos_ex - target_pos_ex).pow(2).sum(0).sqrt().min(1)
            distance_loss = (smooth(displacement_pred[i, :] - distance)).sum()

            """chamfer distance loss"""
            pred_pos_ex = centroids_pred[i, :, :].unsqueeze(2).repeat(1, 1, num_centroids)
            target_pos_ex = single_centroids.unsqueeze(1).repeat(1, num_centroids_pred, 1)
            l1 = (pred_pos_ex - target_pos_ex).pow(2).sum(0).min(0)[0].sum()
            l2 = (pred_pos_ex - target_pos_ex).pow(2).sum(0).min(1)[0].sum()
            chamfer_loss = l1 + l2

            """separation loss"""
            try:
                closet2, _ = (pred_pos_ex - target_pos_ex).pow(2).sum(0).sqrt().squeeze().topk(2, dim=1, largest=False)
                separation_loss = (closet2[:, 0] / closet2[:, 1]).sum()
            except:
                separation_loss = 0

            beta = 0.1
            total_loss += distance_loss + chamfer_loss + beta * separation_loss

        return total_loss
