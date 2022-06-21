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
from torch.nn.utils.rnn import pad_sequence

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


def pad_collate(batch):
    (x, p, t) = zip(*batch)
    x_lens = [len(x0) for x0 in x]

    x_pad = pad_sequence(x, batch_first=True, padding_value=-10.0)
    p_pad = pad_sequence(p, batch_first=True, padding_value=-10.0)
    t_pad = pad_sequence(t, batch_first=True, padding_value=-10.0)

    return x_pad, p_pad, t_pad, x_lens


class Timeseries(Dataset):
    def __init__(self, x, y):
        self.start_idx = []
        self.end_idx = []

        self.p, self.t = zip(*y)
        self.x = x

        self.len = len(x)
        self.sequence_lengths = torch.tensor(list(map(lambda i: i.shape[0], x)))
        self.max_length = max(self.sequence_lengths)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.t[idx]

    def __len__(self):
        return self.len


'''TODO:
We separately train each network with the batch size of 256 using
an Adam [Kingma and Ba 2014] optimizer with a learning rate
lr = 10−3. We follow DIP to train the models for the pose estimation
task using synthetic AMASS first and fine-tune them on DIP-IMU
which contains real IMU measurements. To avoid the vertical drift
due to the error accumulation in the estimation of translations, we
add a gravity velocity 𝑣𝐺 = 0.018 to the Trans-B1 output 𝒗𝑓 to pull
the body down
'''


def train_pose(train_dataset, num_past_frame=20, num_future_frame=5, epoch=20):
    evaluator = PoseEvaluator()
    net = TransPoseNet(num_past_frame, num_future_frame, is_train=True).to(device)

    offline_errs, online_errs = [], []

    loader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=pad_collate)

    for i in range(epoch):
        for x, pose_t, tran_t, seq_lengths in tqdm.tqdm(loader):
            net.reset()
            pose_p_offline, tran_p_offline = net.forward_offline(x, seq_lengths=seq_lengths)
            offline_errs.append(evaluator.eval(pose_p_offline, pose_t))

    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))


def load_dataset(dataset_path, is_train=True, max_length=2000):
    data = torch.load(os.path.join(dataset_path, 'train.pt' if is_train else 'test.pt'))
    xs = [normalize_and_concat(a, r).to(device) for a, r in zip(data['acc'], data['ori'])]
    ys = [(art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t) for p, t in zip(data['pose'], data['tran'])]

    if max_length is not None:
        total_seqs = len(xs)
        # TODO: Slice too long sequences instead of filtering them
        xs = list(filter(lambda x: x.shape[0] < max_length, xs))
        ys = list(filter(lambda x: x[0].shape[0] < max_length, ys))
        print("Filtered {}/{} Sequences that are longer than {} frames".format(total_seqs - len(xs), total_seqs, max_length))

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

    data_list = []
    data_list.append(load_dataset(paths.dipimu_dir))
    # data_list.append(load_dataset(paths.amass_dir))
    # data_list.append(load_dataset(paths.totalcapture_dir))

    sequence_x, sequence_y = merge_dataset(data_list)
    dataset = Timeseries(x=sequence_x, y=sequence_y)
    train_pose(dataset, epoch=20)  # Split train and test data later

    # To make Pose-S2 robust to the prediction errors of leaf-
    # joint positions, during training, we further add Gaussian noise to
    # the leaf-joint positions with 𝜎 = 0.04
    # For Pose-S3, 𝜎 = 0.025
