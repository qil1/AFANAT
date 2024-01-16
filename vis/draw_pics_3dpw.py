import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.AFANAT import *
from utils.opt import Options
from utils.dpw3_3d import Datasets


# single frame 3D
def draw_pic_single(color, mydata, I, J, LR, full_path):
    # num_joints, 3  # x,y,z dimension
    # I
    # J
    # LR

    mydata = mydata[:, [0, 2, 1]]

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    # ax.scatter(x, y, z, c='b')
    # for i in range(len(x)):
    #     ax.text(x[i], y[i], z[i], i, fontsize=2)

    # (250, 40, 40) #FA2828 red
    # (245, 125, 125) #F57D7D pink
    # (11, 11, 11) #0B0B0B black
    # (180, 180, 180) #B4B4B4 gray

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=color)

    # set grid invisible
    ax.grid(None)

    # set X、Y、Z background color white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # set axis invisible
    ax.axis('off')

    plt.savefig(full_path, transparent=True, dpi=300)
    plt.close()


# single frame GT+Pred 3D
def draw_pic_gt_pred(gt, pred, I, J, LR, full_path):
    gt = gt[:, [0, 2, 1]]
    pred = pred[:, [0, 2, 1]]

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    # ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='k', linewidths=1)
    # ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', linewidths=1)

    # (250, 40, 40) #FA2828 red
    # (245, 125, 125) #F57D7D pink
    # (11, 11, 11) #0B0B0B black
    # (180, 180, 180) #B4B4B4 gray

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=1, color='#B4B4B4' if LR[i] else '#B4B4B4')
    for i in np.arange(len(I)):
        x, y, z = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color='#FA2828' if LR[i] else '#FA2828')

    # set grid invisible
    ax.grid(None)

    # set X、Y、Z background color white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # set axis invisible
    ax.axis('off')

    plt.savefig(full_path, transparent=True, dpi=300)
    plt.close()


DRAW_LINE = [
    (0, 3),
    (1, 4),
    (3, 6),
    (6, 9),
    (4, 7),
    (7, 10),
    (2, 1),
    (2, 0),
    (14, 11),
    (11, 8),
    (12, 15),
    (15, 17),
    (17, 19),
    (19, 21),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (5, 2),
    (5, 8),
    (12, 8),
    (13, 8)
]
I, J, LR = [], [], []
for i in range(len(DRAW_LINE)):
    I.append(DRAW_LINE[i][0])
    J.append(DRAW_LINE[i][1])
    LR.append(1)

if __name__ == "__main__":
    config = Options().parse()

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = Datasets(config, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    model = get_model(config, device)
    model.to(device)

    cp_path = config.model_path % (config.save_dir_name, config.iter)
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    model.load_state_dict(model_cp['model_dict'])

    dim_used = test_dataset.dim_used
    cnt = 0
    for (gt3d) in test_loader:
        cnt += 1
        print(cnt)
        gt3d = gt3d.type(dtype).to(device).contiguous()
        gt3d /= 1000.
        batch_size, seq_n, _ = gt3d.shape

        condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d', d=3).clone()
        pred32 = gt3d.clone()
        _, dec_res = model(
            x=condition,
        )
        dec_res = rearrange(dec_res, 't b c d -> b t (c d)')
        pred32[:, config.t_his:config.t_his + config.t_pred, dim_used] = dec_res[:, :config.t_pred]
        pred32 = pred32[:, :, dim_used].reshape([-1, config.t_his + config.t_pred, 23, 3])
        gt3d_t = rearrange(gt3d[:, :, dim_used], 'b t (c d) -> b t c d', d=3).contiguous()

        sample_gt = gt3d_t[0].detach().cpu()*1000
        sample_pred = pred32[0].detach().cpu()*1000
        t, c, d = sample_gt.shape

        for t_id in range(t):
            if not os.path.exists('./vis/{}'.format("3dpw_" + str(cnt))):
                os.mkdir('./vis/{}'.format("3dpw_" + str(cnt)))
            draw_pic_single('#B4B4B4', sample_gt[t_id], I, J, LR, './vis/{}/gt_{}.png'.format("3dpw_" + str(cnt), t_id))
            draw_pic_single('#FA2828', sample_pred[t_id], I, J, LR,
                            './vis/{}/pred_{}.png'.format("3dpw_" + str(cnt), t_id))
            # draw_pic_gt_pred(sample_gt[t_id], sample_pred[t_id], I, J, LR,
            #                  './vis/{}/{}.png'.format(act + str(cnt), t_id))

        if cnt == 20:
            break
