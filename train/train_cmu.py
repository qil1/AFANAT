import os
import numpy as np
import sys
import pickle
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from models.AFANAT import *
from utils.opt import Options
from utils.log import save_csv_eval_log
from utils.logger import create_logger
from utils.torch import to_cpu
from utils.CMU_motion_3d import CMU_Motion3D, ACTION
from utils.util import cal_total_model_param, cal_mpjpe_every_frame, seed_torch


def loss_function(joint_pred_lst, joint_pred, joint_gt):
    loss = 0
    losses = []
    if len(joint_pred_lst) > 1:
        loss = torch.linalg.norm(
            rearrange(joint_pred, 't b c d -> (b t c) d') -
            rearrange(joint_gt[:, config.t_his:, ], 'b t (c d) -> (b t c) d', d=3),
            ord=2, axis=1).mean()
        losses = [loss.item()]

    for i, t_pred_ in enumerate(config.t_pred_lst):
        loss_ = torch.linalg.norm(
            rearrange(joint_pred_lst[i], 't b c d -> (b t c) d') -
            rearrange(joint_gt[:, config.t_his:, ] if t_pred_ < config.t_pred else joint_gt, 'b t (c d) -> (b t c) d',
                      d=3),
            ord=2, axis=1).mean()
        loss += loss_
        losses.append(loss_.item())

        # strong short-term constraint
        if t_pred_ < config.t_pred and config.f1_weight:
            loss += config.f1_weight * torch.linalg.norm(
                rearrange(joint_pred_lst[i][:t_pred_], 't b c d -> (b t c) d') -
                rearrange(joint_gt[:, config.t_his:config.t_his + t_pred_], 'b t (c d) -> (b t c) d', d=3),
                ord=2, axis=1).mean()

        if t_pred_ * 2 < config.t_pred and config.f2_weight:
            loss += config.f2_weight * torch.linalg.norm(
                rearrange(joint_pred_lst[i][:t_pred_ * 2], 't b c d -> (b t c) d') -
                rearrange(joint_gt[:, config.t_his:config.t_his + t_pred_ * 2], 'b t (c d) -> (b t c) d', d=3),
                ord=2, axis=1).mean()

    return loss, np.array([loss.item()] + losses)  #


def train_func(epoch):
    model.train()

    train_losses = np.array(([0.0, 0.0] if len(config.t_pred_lst) > 1 else [0.0]) + [0.0 for _ in config.t_pred_lst])
    loss_names = (['TOTAL', 'Ensemble'] if len(config.t_pred_lst) > 1 else ['TOTAL']) + \
                 [f'Branch_{t_pred_}' for t_pred_ in config.t_pred_lst]
    total_num_sample = 0
    dim_used = dataset.dim_used
    st = time.time()
    with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
        for gt3d in tqdmDataLoader:
            gt3d = gt3d.type(dtype).to(device).contiguous()
            gt3d /= 1000.
            condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d', d=3).clone()
            # gt = gt3d[:, config.t_his:, dim_used].clone()
            gt = gt3d[:, :, dim_used].clone()
            out_res_lst, out_res = model(
                x=condition,
            )
            loss, losses = loss_function(out_res_lst, out_res, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += losses
            total_num_sample += 1

            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch": epoch,
                "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                "loss": loss.item()
            })

    dt = time.time() - st
    scheduler.step()
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.6f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('train_' + name, loss, epoch)


def val_func(epoch, test=False):
    model.eval()
    action_mpjpe = {}
    for action in ACTION:
        action_mpjpe[action] = [0. for i in range(25)]

    dim_used = dataset.dim_used
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    if not test:
        raise NotImplementedError("there is no val_data_loader")
    else:
        loader = test_data_loader

    total_time = 0
    cnt = 0
    for act in ACTION:
        total_num_sample = 0
        print("Action: ", act)
        pred_mpjpe_all = np.zeros([config.t_pred])
        for (gt3d) in tqdm(loader[act]):
            gt3d = gt3d.type(dtype).to(device).contiguous()
            gt3d /= 1000.
            batch_size, seq_n, _ = gt3d.shape
            condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d',
                                  d=3).clone()
            pred32 = gt3d.clone()[:, config.t_his:config.t_his + config.t_pred]

            t_s = time.time()
            _, dec_res = model(
                x=condition,
            )
            total_time += (time.time() - t_s)
            cnt += 1

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

    avg_time = total_time / cnt
    print("avg forward pass time:", avg_time)

    avg_mpjpe1, avg_mpjpe2, avg_mpjpe3, avg_mpjpe4, avg_mpjpe5, avg_mpjpe6 = 0., 0., 0., 0., 0., 0.
    losses_str = ['' for i in range(11)]
    losses_str[0] = losses_str[0].join(
        "action | mpjpe 2frame | mpjpe 4frame | mpjpe 8frame | mpjpe 10frame | mpjpe 14frame | mpjpe 25frame")
    for i in range(8):  # 8 actions
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
    losses_str[10] = losses_str[10].join("====================================================")
    logger_val.info('====> Epoch: {} ({})'.format(epoch, "val" if not test else "test"))
    for i in range(11):
        logger_val.info('{}'.format(losses_str[i]))

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
        save_csv_eval_log(config, head, ret_log, is_create=is_create)
        is_create = False
    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    save_csv_eval_log(config, head, write_ret_log, is_create=False)


if __name__ == "__main__":
    '''setup'''
    config = Options().parse()
    seed_torch(config.seed)
    config.log_dir = config.log_dir % config.save_dir_name
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    config.tb_dir = config.tb_dir % config.save_dir_name
    if not os.path.exists(config.tb_dir):
        os.mkdir(config.tb_dir)
    config.csv_dir = config.csv_dir % config.save_dir_name
    if not os.path.exists(config.csv_dir):
        os.mkdir(config.csv_dir)
    save_dir = os.sep.join(config.model_path.split('/')[:-1]) % config.save_dir_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sorted(config.t_pred_lst)
    if config.t_pred_lst[0] < 1 or config.t_pred_lst[-1] > config.t_pred:
        raise RuntimeError("invalid t_pred_lst")
    if config.t_pred_lst[-1] == config.t_pred:
        config.t_pred_lst = [config.t_pred] + config.t_pred_lst[:-1]

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_index)
    tb_logger = SummaryWriter(config.tb_dir)
    logger = create_logger(os.path.join(config.log_dir, 'log_train.txt'))
    logger_val = create_logger(os.path.join(config.log_dir, 'log_val.txt'))
    print(device)
    print(config.num_epoch)

    '''data'''
    dataset = CMU_Motion3D(config, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_data_loader = {}
    test_tot = 0
    for act in ACTION:
        dataset = CMU_Motion3D(opt=config, split=1, actions=act)
        test_tot += dataset.__len__()
        test_data_loader[act] = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=0,
                                           pin_memory=True)
    print('>>> Testing dataset length: {:d}'.format(test_tot))

    '''model'''
    model = get_model(config, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.5)
    if config.iter > 0:
        cp_path = config.model_path % (config.save_dir_name, config.iter - 1)
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    model.to(device)
    model.train()
    cal_total_model_param([model])

    for i in range(config.iter, config.num_epoch):
        print("epoch:", i)
        train_func(i)
        with torch.no_grad():
            val_func(i, test=True)
        if config.save_model_interval > 0 and (i + 1) % config.save_model_interval == 0:
            with to_cpu(model):
                cp_path = config.model_path % (config.save_dir_name, i + 1)
                model_cp = {'model_dict': model.state_dict()}
                pickle.dump(model_cp, open(cp_path, 'wb'))
