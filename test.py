import argparse
import os
from data_utils.toothDataLoader import toothDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from visualizer.helper_data_plot import Plot as Plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['gingiva', 'tooth1', 'tooth2', 'tooth3', 'tooth4', 'tooth5', 'tooth6', 'tooth7', 'tooth8',
           'tooth9', 'tooth10', 'tooth11', 'tooth12', 'tooth13', 'tooth14', 'x']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='2022-04-25_14-03', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=1, help='area for testing, option: 1-6 [default: 1]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/tooth/maxillary/'
    NUM_CENTROIDS = 14
    NUM_CLASSES = 16
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    TEST_DATASET_WHOLE_SCENE = toothDatasetWholeScene(root, split='test', test_area=args.test_area,
                                                      block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CENTROIDS).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_pred_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_pred_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))  # 147456, 16

            scene_data, scene_label, scene_smpw, scene_point_index, coords_max = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            sbatch_centroids = np.zeros((s_batch_num, BATCH_SIZE, NUM_CENTROIDS, 3))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]   # 16, 4096
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)

                target = torch.Tensor(batch_label)
                target = target.int().cuda()

                seg_pred, proposal_index = classifier(torch_data, target)  # B, num_centroids, n, 2 / B, num_centroids, n
                # 16,14,1024,16  16,14,1024

                points2 = torch_data[:, :3, :]  # 16,3,4096
                from models.tsegnet_utils import gather_points
                seg_proposal = np.argmax(seg_pred.contiguous().cpu().data, 3)   # 16,14,1024
                batch_point_index_tensor = torch.Tensor(batch_point_index).float().cuda().unsqueeze(1)
                index_proposal = gather_points(batch_point_index_tensor, proposal_index).permute(0, 1, 3, 2)
                index_vote = index_proposal.reshape(BATCH_SIZE * NUM_CENTROIDS, -1)
                seg_pred_vote = seg_proposal.reshape(BATCH_SIZE * NUM_CENTROIDS, -1)
                vote_label_pool = add_vote(vote_label_pool, index_vote, seg_pred_vote, None)
                points_proposal = gather_points(points2, proposal_index).permute(0, 1, 3, 2)
                # target_plot = seg_proposal.numpy()[:, :, :]
                # points_plot = points_proposal.numpy()
                # target_plot2 = np.squeeze(target_plot.reshape(-1, 1))
                # points_plot2 = points_plot.reshape(-1, 3)

            pred_label = np.argmax(vote_label_pool, 1)
            from visualizer.helper_data_plot import Plot as Plot
            # Plot.draw_pc_semins(pc_xyz=whole_scene_data[:, :3], pc_semins=pred_label)
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_pred_class_tmp[l] += np.sum((pred_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_pred_class[l] += total_pred_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            dsc_map = np.array(total_correct_class_tmp*2) / \
                      (np.array(total_iou_deno_class_tmp + total_correct_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
            dsc = np.array(total_correct_class*2) / (np.array(total_iou_deno_class+total_correct_class, dtype=np.float) + 1e-6)
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            log_string(iou_per_class_str)
            log_string('eval point avg class IoU: %f' % np.mean(IoU))
            log_string('eval point avg class dsc: %f' % np.mean(dsc))
            log_string('eval whole scene point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            log_string('eval whole scene point accuracy: %f' % (
                    np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

            print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)











