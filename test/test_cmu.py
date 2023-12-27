import os
import numpy as np
import sys
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.AFANAT import *
from utils.opt import Options
from utils.log import save_csv_eval_log
from utils import CMU_motion_3d as CMU_Motion3D
from utils.CMU_motion_3d import ACTION
from utils.util import cal_mpjpe_every_frame,seed_torch


def val_func():
    model.eval()
    action_mpjpe = {}
    for action in ACTION:
        action_mpjpe[action] = [0. for i in range(25)]
    dim_used = dataset.dim_used
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
    for act in ACTION:
        total_num_sample = 0
        print("Action: ", act)
        pred_mpjpe_all = np.zeros([config.t_pred])
        for (gt3d) in tqdm(test_data_loader[act]):
            gt3d = gt3d.type(dtype).to(device).contiguous()
            gt3d /= 1000.
            batch_size, seq_n, _ = gt3d.shape
            condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d',
                                  d=3).clone()

            pred32 = gt3d.clone()[:, config.t_his:config.t_his + config.t_pred]

            _, dec_res = model(
                x=condition,
            )

            dec_res = rearrange(dec_res, 't b c d -> b t (c d)')

            pred32[:, :, dim_used] = dec_res[:, :config.t_pred]
            pred32[:, :, index_to_ignore] = pred32[:, :, index_to_equal]
            pred32 = pred32.reshape([-1, config.t_pred, 38, 3])
            gt3d_t = rearrange(gt3d[:, config.t_his:config.t_his + config.t_pred], 'b t (c d) -> b t c d',
                               d=3).contiguous()
            mpjpe_total_batch_sum = cal_mpjpe_every_frame(gt3d_t, pred32)
            pred_mpjpe_all += mpjpe_total_batch_sum.cpu().data.numpy()
            #
            total_num_sample += batch_size

        mpjpe_ret_dict = pred_mpjpe_all / total_num_sample
        for t_need in range(25):
            action_mpjpe[act][t_need] = mpjpe_ret_dict[t_need]

    avg_mpjpe1, avg_mpjpe2, avg_mpjpe3, avg_mpjpe4, avg_mpjpe5, avg_mpjpe6 = 0., 0., 0., 0., 0., 0.
    losses_str = ['' for i in range(11)]
    losses_str[0] = losses_str[0].join(
        "action | mpjpe 2frame | mpjpe 4frame | mpjpe 8frame | mpjpe 10frame | mpjpe 14frame | mpjpe 25frame")
    for i in range(8):
        avg_mpjpe1 += action_mpjpe[ACTION[i]][1]
        avg_mpjpe2 += action_mpjpe[ACTION[i]][3]
        avg_mpjpe3 += action_mpjpe[ACTION[i]][7]
        avg_mpjpe4 += action_mpjpe[ACTION[i]][9]
        avg_mpjpe5 += action_mpjpe[ACTION[i]][13]
        avg_mpjpe6 += action_mpjpe[ACTION[i]][24]
        losses_str[i + 1] = losses_str[i + 1].join(
            "{} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f}".format(ACTION[i],
                                                                                          action_mpjpe[ACTION[i]][1],
                                                                                          action_mpjpe[ACTION[i]][3],
                                                                                          action_mpjpe[ACTION[i]][7],
                                                                                          action_mpjpe[ACTION[i]][9],
                                                                                          action_mpjpe[ACTION[i]][13],
                                                                                          action_mpjpe[ACTION[i]][24]))

    losses_str[9] = losses_str[9].join(
        "{} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f}".format("Average", avg_mpjpe1 / 8.,
                                                                                      avg_mpjpe2 / 8., avg_mpjpe3 / 8.,
                                                                                      avg_mpjpe4 / 8., avg_mpjpe5 / 8.,
                                                                                      avg_mpjpe6 / 8.))
    losses_str[10] = losses_str[10].join("====================================================\n")
    for i in range(11):
        print('{}'.format(losses_str[i]))

    # csv save
    is_create = True
    avg_ret_log = []
    for act in ACTION:
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in range(len(action_mpjpe['basketball'])):
            ret_log = np.append(ret_log, [action_mpjpe[act][k]])
            head = np.append(head, ['test_' + str((k + 1) * 40)])

        avg_ret_log.append(ret_log[1:])
        save_csv_eval_log(config, head, ret_log, is_create=is_create, file_name='test_'+cp_path.split('/')[-1].split('.')[0])

        is_create = False
    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    save_csv_eval_log(config, head, write_ret_log, is_create=False, file_name='test_'+cp_path.split('/')[-1].split('.')[0])


if __name__ == "__main__":
    config = Options().parse()
    config.csv_dir = config.csv_dir % config.save_dir_name
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_index)

    test_data_loader = {}
    for act in ACTION:
        dataset = CMU_Motion3D.CMU_Motion3D(opt=config, split=2, actions=act)
        test_data_loader[act] = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=0,
                                          pin_memory=True)
    model = get_model(config, device)

    model.to(device)

    if config.iter > 0:
        cp_path = config.model_path % (config.save_dir_name, config.iter)
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

        with torch.no_grad():
            val_func()

    else:
        for iter in range(191, 200 + 1):
            cp_path = config.model_path % (config.save_dir_name, iter)
            print('loading model from checkpoint: %s' % cp_path)
            model_cp = pickle.load(open(cp_path, "rb"))
            model.load_state_dict(model_cp['model_dict'])

            with torch.no_grad():
                val_func()
