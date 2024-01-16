from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

ACTION = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
          "washwindow"]

'''
adapted from https://github.com/705062791/PGBIG
'''


class CMU_Motion3D(Dataset):
    # frame rate is not 25hz, but 30hz / 2 = 15hz
    # (see "https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics/issues/4"),
    # but to maintain consistency with the cited papers, we also employ "25Hz" in the table1 in our paper.
    def __init__(self, opt, split, actions='all'):

        self.path_to_data = opt.data_dir
        input_n = opt.t_his
        output_n = opt.t_pred

        self.split = split
        is_all = actions
        actions = data_utils.define_actions_cmu(actions)
        # actions = ['walking']
        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = True

        if not is_test:
            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
                                                                            input_n, output_n,
                                                                            is_test=is_test)
        else:
            # all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_all(opt, path_to_data, actions,
            #                                                                 input_n, output_n,
            #                                                                 is_test=is_test)

            all_seqs, dim_ignore, dim_use = data_utils.load_data_cmu_3d_n(opt, path_to_data, actions,
                                                                          input_n, output_n,
                                                                          is_test=is_test)

        self.all_seqs = all_seqs
        self.dim_used = dim_use

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return self.all_seqs[item]
