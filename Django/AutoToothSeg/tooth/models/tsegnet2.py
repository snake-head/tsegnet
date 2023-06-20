import torch.nn as nn
import torch.nn.functional as F
from tsegnet_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, get_centroids, \
    PointNetFeaturePropagation, get_proposal
import torch
import numpy as np


class get_model(nn.Module):
    def __init__(self, num_centroids):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [2.5, 5], [16, 32], 9 + 3, [[9, 32, 32], [9, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [5, 10], [16, 32], 64 + 3, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [10, 20], [16, 32], 256 + 3, [[256, 196, 256], [256, 196, 256]])
        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 9, [64, 32])
        self.conv1 = nn.Conv1d(36, 36, 1)
        self.bn1 = nn.BatchNorm1d(36)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(36, 16, 1)

        self.displacement = nn.Sequential(
            nn.Conv1d(515, 256, (1,)),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 64, (1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 4, (1,)),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
        )

        self.centroid = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_centroids * 3),
        )

        self.num_centroids = num_centroids

    def forward(self, xyz, centroids):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l3 = torch.cat((l3_xyz, l3_points), 1)
        # displacement = self.displacement(l3)
        # distance = displacement[:, 3, :]

        # displacement = torch.cat((l3_xyz, displacement), 1)
        # displacement = displacement.view(displacement.size(0), -1)
        # centroids = self.centroid(displacement)
        # centroids = centroids.view(centroids.size(0), 3, self.num_centroids)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        centroids_tensor = torch.from_numpy(centroids.reshape(-1, 14, 3)).permute(0, 2, 1)
        proposals, proposal_index = get_proposal(l0_xyz, l0_points, centroids_tensor.float().cuda())  # B, 14, 3+32+1, n / B, 14, n
        proposals = proposals.view(-1, proposals.shape[-2], proposals.shape[-1])    # B*14, 36, n
        x = self.drop1(F.relu(self.bn1(self.conv1(proposals))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1).reshape(l0_points.shape[0], -1, proposals.shape[-1], 16)  # B, 14, n, 16

        return x, proposal_index


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
