import torch
import numpy as np
from models.tsegnet_utils import get_centroids

t_net = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
origin_points = torch.randn(16, 16384, 9)
a = origin_points.reshape(1, -1, 9)
b = a.numpy()
target = ((torch.rand(16, 16384)) * 10).trunc()
# a = get_centroids(origin_points, target)

# arr = np.array([1, 2, 3, 4])
# print(arr[-3:])
# pred_distance = torch.randn(16, 200)
# subsample_pos = torch.randn(16, 3, 200)
# pred_pos = torch.randn(16, 3, 12)
# target_pos = torch.randn(16, 3, 11)
# a = torch.randn(16, 256)
# a = a.reshape(-1, 1).squeeze()
displacement_pred = torch.randn(16, 256)
centroids_pred = torch.randn(16, 3, 12)
subsampled_points = torch.randn(16, 3, 256)
# displacement_pred_ex = displacement_pred.reshape(-1, 1).squeeze()
# displacement_pred_ex = torch.where(displacement_pred_ex > 0.2, torch.tensor(1.), torch.tensor(0.))
# index = list()
# for j, item in enumerate(displacement_pred_ex):
#     if torch.equal(item, torch.tensor(0.)):
#         index.append(j)
# index_np = np.array(index)
# index_tensor = torch.from_numpy(index_np)
# displacement_pred_ex = torch.gather(displacement_pred_ex, dim=0, index=index_tensor.long())
# subsampled_points_ex = subsampled_points.reshape(-1, 3).gather(dim=0, index=index_tensor.long().unsqueeze(1).repeat(1, 3))

"""
pred_distance:  [B, num of effective subsampled points]
subsample_pos:  [B, 3, num of effective subsampled points]
pre_pos: [B, 3, 12]
target_pos: [B, 3, num of centroids]
"""

centroids_gt = get_centroids(origin_points, target)
num_points = displacement_pred.size(1)
batch_num = displacement_pred.size(0)
total_loss = 0


def smooth(x):
    x_abs = x.abs()
    smooth1 = torch.where(x_abs < 1, 0.5 * torch.pow(x, 2), torch.abs(x) - 0.5)
    return smooth1
