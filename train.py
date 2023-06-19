import argparse
import os
from data_utils.toothDataLoader import toothDataset
from models.tsegnet_utils import evalue_distance
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='tsegnet', help='model name [default: tsegnet]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='2022-03-15_17-25', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=1, help='Which area to use for test, option: 1-6 [default: 1]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/tooth/maxillary/'
    NUM_CENTROIDS = 14
    NUM_CLASSES = 16
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = toothDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area,
                                 block_size=50.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = toothDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area,
                                block_size=50.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/tsegnet_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CENTROIDS).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    best_correct = 100
    best_loss = 1000000

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target, coord_max) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                   smoothing=0.9):
            optimizer.zero_grad()

            B, N = target.shape
            points = points.data.numpy()

            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            origin_points = points.transpose(2, 1)  # B, 9, N
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            # centroids_pred, displacement_pred, subsampled_points = classifier(points)
            seg_pred, proposal_index = classifier(points, target)  # B, num_centroids, n, 2 / B, num_centroids, n
            target2 = torch.zeros([B, 2, N])
            target2[:, 0, :] = target
            target2[:, 1, :] = 1 - target2[:, 0, :]
            points2 = points[:, :3, :]
            from models.tsegnet_utils import gather_points
            target_proposal = gather_points(target2.long().cuda(), proposal_index).permute(0, 1, 3,
                                                                                           2)  # 16, 14, 1024, 2
            seg_proposal = np.argmax(seg_pred.contiguous().cpu().data, 3)
            points_proposal = gather_points(points2, proposal_index).permute(0, 1, 3, 2)
            target_plot = seg_proposal.numpy()[:, :, :]
            points_plot = points_proposal.numpy()
            # for k in range(14):
            #     target_plot[:, k, :] *= (k + 1)
            target_plot2 = target_plot.reshape(16, -1)
            points_plot2 = points_plot.reshape(16, -1, 3)
            from visualizer.helper_data_plot import Plot as Plot
            Plot.draw_pc_semins(pc_xyz=points_plot2[0, :, :], pc_semins=target_plot2[0, :])
            # for j in range(14):
            #     Plot.draw_pc_semins(pc_xyz=points_plot[0, j, :], pc_semins=target_plot[0, j, :])

            target = target_proposal[:, :, :, 0].reshape(-1, 1)[:, 0]
            target2 = target_proposal[:, :, :, 0].reshape(16, -1, 1)[:, :, 0]
            batch_label = target2.cpu().data.numpy()
            seg_pred = seg_pred.reshape(-1, 16)
            seg_pred2 = seg_pred.reshape(16, -1, 16)

            pred_val = seg_pred2.contiguous().cpu().data.numpy()
            target = target.long().cuda()
            loss = criterion(seg_pred, target, None)
            loss.backward()
            optimizer.step()

            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            # coord_max = coord_max.unsqueeze(2).repeat(1, 1, NUM_CENTROIDS)
            # coord_max = coord_max.float().cuda()
            # centroids_pred *= coord_max

            """remove far subsampled points"""
            # displacement_pred_ex = displacement_pred.reshape(-1, 1).squeeze()
            # displacement_pred_ex = torch.where(displacement_pred_ex > 0.2, torch.tensor(1.), torch.tensor(0.))
            # index = list()
            # for j, item in enumerate(displacement_pred_ex):
            #     if torch.equal(item, torch.tensor(0.)):
            #         index.append(j)
            # index_np = np.array(index)
            # index_tensor = torch.from_numpy(index_np)
            # displacement_pred_ex = torch.gather(displacement_pred_ex, dim=0, index=index_tensor.long())
            # subsampled_points_ex = subsampled_points.reshape(-1, 3)\
            #     .gather(dim=0, index=index_tensor.long().unsqueeze(1).repeat(1, 3))
            # centroids_pred_ex = centroids_pred.reshape

            # loss = criterion(displacement_pred, subsampled_points, centroids_pred, origin_points, target)

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            # correct = evalue_distance(centroids_pred, origin_points, target)
            # from models.tsegnet_utils import get_centroids
            # centroids_gt = get_centroids(origin_points.transpose(1, 2).reshape(1, -1, 9), target.reshape(1, -1))
            # centroids_pred_np = centroids_pred.cpu().detach().numpy()
            # centroids_gt_tensor = torch.stack(centroids_gt[0])
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''

        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target, coord_max) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                       smoothing=0.9):

                B, N = target.shape

                origin_points = points.transpose(2, 1)
                points = points.data.numpy()
                # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, proposal_index = classifier(points, target)  # B, num_centroids, n, 2 / B, num_centroids, n
                target2 = torch.zeros([B, 2, N])
                target2[:, 0, :] = target
                target2[:, 1, :] = 1 - target2[:, 0, :]
                from models.tsegnet_utils import gather_points
                target_proposal = gather_points(target2.long().cuda(), proposal_index).permute(0, 1, 3, 2)
                # B, num_centroids, n, 2
                target = target_proposal[:, :, :, 0].reshape(-1, 1)[:, 0]
                target2 = target_proposal[:, :, :, 0].reshape(16, -1, 1)[:, :, 0]
                batch_label = target2.cpu().data.numpy()
                seg_pred = seg_pred.reshape(-1, 16)
                seg_pred2 = seg_pred.reshape(16, -1, 16)
                # coord_max = coord_max.unsqueeze(2).repeat(1, 1, NUM_CENTROIDS)
                # coord_max = coord_max.float().cuda()
                # centroids_pred *= coord_max

                pred_val = seg_pred2.contiguous().cpu().data.numpy()
                target = target.long().cuda()
                loss = criterion(seg_pred, target, None)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                # correct = evalue_distance(centroids_pred, origin_points, target)
                # from models.tsegnet_utils import get_centroids
                # centroids_gt = get_centroids(origin_points.transpose(1, 2), target)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                # tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                # labelweights += tmp
                #
                for l in range(2):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            #
            # iou_per_class_str = '------- IoU --------\n'
            # for l in range(NUM_CLASSES):
            #     iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
            #         total_correct_class[l] / float(total_iou_deno_class[l]))
            #
            # log_string(iou_per_class_str)
            # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            # log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            # if total_correct / float(total_seen) <= best_correct:
            #     best_correct = total_correct / float(total_seen)
            #     logger.info('Save model...')
            #     savepath = str(checkpoints_dir) + '/best_model.pth'
            #     log_string('Saving at %s' % savepath)
            #     state = {
            #         'epoch': epoch,
            #         'mean_distance': best_correct,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
            #     log_string('Saving model....')
            # log_string('Best distance: %f' % best_correct)

            if (loss_sum / num_batches) <= best_loss:
                best_loss = loss_sum / num_batches
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'mean_loss': best_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best distance: %f' % best_loss)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
