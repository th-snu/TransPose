r"""
    train the pose estimation.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
from net import TransPoseNet
from config import *
import os
import articulate as art
from utils import normalize_and_concat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


class Timeseries(Dataset):
    def __init__(self, x, y):
        self.start_idx = []
        self.end_idx = []

        self.xs = torch.vstack(x)
        pose, tran = zip(*y)
        self.sequence_lengths = torch.tensor(list(map(lambda i: i.shape[0], x)))
        self.max_length = max(self.sequence_lengths)
        self.end_idx = self.sequence_lengths.cumsum(dim=-1)
        self.len = self.end_idx.shape[0]

        self.pose = torch.vstack(pose)
        self.tran = torch.vstack(tran)

    def __getitem__(self, idx):
        sequence_slice = slice(self.end_idx[idx-1] if idx != 0 else 0, self.end_idx[idx])
        return self.xs[sequence_slice], self.pose[sequence_slice], self.tran[sequence_slice]

    def __len__(self):
        return self.len


def train_pose(train_dataset, num_past_frame=20, num_future_frame=5, epoch=20):
    evaluator = PoseEvaluator()
    net = TransPoseNet(num_past_frame, num_future_frame, is_train=False).to(device)

    offline_errs, online_errs = [], []

    loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    for i in range(epoch):
        for x, y in tqdm.tqdm(loader):
            net.reset()
            online_results = [net.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
            pose_p_online, tran_p_online = [torch.stack(_)[num_future_frame:] for _ in zip(*online_results)]
            pose_p_offline, tran_p_offline = net.forward_offline(x)
            pose_t, tran_t = y
            offline_errs.append(evaluator.eval(pose_p_offline, pose_t))
            online_errs.append(evaluator.eval(pose_p_online, pose_t))

    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    print('============== online ================')
    evaluator.print(torch.stack(online_errs).mean(dim=0))


def load_dataset(dataset_path, is_train=True):
    data = torch.load(os.path.join(dataset_path, 'train.pt' if is_train else 'test.pt'))
    xs = [normalize_and_concat(a, r).to(device) for a, r in zip(data['acc'], data['ori'])]
    ys = [(art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t) for p, t in zip(data['pose'], data['tran'])]
    return xs, ys


def merge_dataset(datasets):
    # Merge sequences from datasets into one list
    xs, ys = [], []
    for x, y in datasets:
        xs.extend(x)
        ys.extend(y)
    return xs, ys


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False   # if cudnn error, uncomment this line

    dipimu_data = load_dataset(paths.dipimu_dir)
    amass_data = load_dataset(paths.amass_dir)
    # load_dataset(paths.totalcapture_dir)

    sequence_x, sequence_y = merge_dataset([dipimu_data, amass_data])
    dataset = Timeseries(x=sequence_x, y=sequence_y)
    train_pose(dataset, epoch=20)  # Split train and test data later

    # To make Pose-S2 robust to the prediction errors of leaf-
    # joint positions, during training, we further add Gaussian noise to
    # the leaf-joint positions with 𝜎 = 0.04
    # For Pose-S3, 𝜎 = 0.025
