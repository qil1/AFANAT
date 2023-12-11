import os
import numpy as np
import sys
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.AFANAT import *
from utils.opt import Options
from utils.dpw3_3d import Datasets
from utils.log import save_csv_eval_log
from utils.util import cal_mpjpe_every_frame, seed_torch


def val_func():
    model.eval()

    dim_used = test_dataset.dim_used

    total_num_sample = 0
    pred_mpjpe_all = np.zeros([config.t_pred])
    for (gt3d) in tqdm(test_loader):
        gt3d = gt3d.type(dtype).to(device).contiguous()
        gt3d /= 1000.
        batch_size, seq_n, _ = gt3d.shape
        condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d', d=3).clone()

        pred32 = gt3d.clone()[:, config.t_his:config.t_his + config.t_pred]

        _, dec_res = model(
                x=condition,
        )

        dec_res = rearrange(dec_res, 't b c d -> b t (c d)')
        pred32[:, :, dim_used] = dec_res[:, :config.t_pred]
        pred32 = pred32.reshape([-1, config.t_pred, 24, 3])
        gt3d_t = rearrange(gt3d[:, config.t_his:config.t_his + config.t_pred], 'b t (c d) -> b t c d',
                           d=3).contiguous()
        mpjpe_total_batch_sum = cal_mpjpe_every_frame(gt3d_t, pred32)
        pred_mpjpe_all += mpjpe_total_batch_sum.cpu().data.numpy()
        #
        total_num_sample += batch_size

    mpjpe_ret_dict = pred_mpjpe_all / total_num_sample

    avg_mpjpe1, avg_mpjpe2, avg_mpjpe3, avg_mpjpe4, avg_mpjpe5 = 0., 0., 0., 0., 0.
    avg_mpjpe1 += mpjpe_ret_dict[4]
    avg_mpjpe2 += mpjpe_ret_dict[9]
    avg_mpjpe3 += mpjpe_ret_dict[14]
    avg_mpjpe4 += mpjpe_ret_dict[19]
    avg_mpjpe5 += mpjpe_ret_dict[24]

    losses_str = ['' for i in range(3)]
    # print("action | mpjpe 2frame | mpjpe 10frame | mpjpe 25frame")
    losses_str[0] = losses_str[0].join(
        "action | mpjpe 5frame | mpjpe 10frame | mpjpe 15frame | mpjpe 20frame | mpjpe 25frame")

    losses_str[1] = losses_str[1].join(
        "{} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f}".format("Average", avg_mpjpe1,
                                                                                      avg_mpjpe2, avg_mpjpe3,
                                                                                      avg_mpjpe4, avg_mpjpe5))
    losses_str[2] = losses_str[2].join("====================================================\n")

    # csv save
    ret_log = np.array(['avg'])
    head = np.array(['action'])
    for k in range(len(mpjpe_ret_dict)):
        ret_log = np.append(ret_log, [mpjpe_ret_dict[k]])
        head = np.append(head, ['test_' + str((k + 1) * 40)])
    save_csv_eval_log(config, head, ret_log, is_create=True, file_name='test_'+cp_path.split('/')[-1].split('.')[0])


if __name__ == "__main__":
    config = Options().parse()
    config.csv_dir = config.csv_dir % config.save_dir_name
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_index)

    test_dataset = Datasets(config, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    model = get_model(config, device)

    model.to(device)

    for iter in range(191, 200 + 1):
        cp_path = config.model_path % (config.save_dir_name, iter)
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

        with torch.no_grad():
            val_func()
