from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch
from scipy import signal
import os

'''
adapted from https://github.com/705062791/PGBIG
'''

ACTION = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown',
          'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']

dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                     26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                     75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])


class Datasets(Dataset):
    def __init__(self, opt, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.opt = opt
        self.path_to_data = opt.data_dir
        self.split = split
        self.in_n = opt.t_his
        self.out_n = opt.t_pred
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]], dtype=object)
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = [actions]
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1 or opt.test_sample_num < 0:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().to(self.opt.gpu_index)
                        # remove global translation and rotation
                        the_sequence[:, 0:6] = 0  # 0:6
                        p3d = data_utils.expmap2xyz_torch(self.opt, the_sequence)
                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                        valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)
                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        tmp_action = [ACTION.index(action)] * len(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_action))
                        key += 1
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().to(self.opt.gpu_index)
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(self.opt, the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().to(self.opt.gpu_index)
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(self.opt, the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    # [n, 35]
                    fs_sel1, fs_sel2 = data_utils.find_indices_n(num_frames1, num_frames2, seq_len,
                                                                 input_n=self.in_n,
                                                                 test_sample_num=opt.test_sample_num)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_action = [ACTION.index(action)] * len(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_action))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    tmp_action = [ACTION.index(action)] * len(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2, tmp_action))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])  # np.array([11, 16, 20, 23, 24, 28, 31,4,5,9,10,21,22,29,30])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame, action_idx = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        # [20, 96]
        src = self.p3d[key][fs]  # num_frames, joints*3
        action_idx = torch.tensor(action_idx, dtype=torch.int64)
        return src, action_idx


if __name__ == '__main__':
    from utils.opt import Options

    opt = Options().parse()
    opt.test_sample_num = 8

    data_set = Datasets(opt, split=2)
    data_set_len = len(data_set)
    data_list = []

    for i in data_set:
        data_list.append(i)
    data_list = np.array(data_list)
    np.save('./complete_data.npy', data_list)
    print(1)
