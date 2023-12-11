import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())
from utils.opt import Options
from models.AFANAT import *


def vis_save_mat(mat, figsize=(15, 2), filename='fusion_weight'):
    # mat = (mat / np.sum(mat, 0))
    df = pd.DataFrame(mat)

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df, cmap=plt.cm.Blues, linewidths=0.5, annot=True)

    # Decorations
    col_label = ["T{}".format(i) for i in range(1, mat.shape[1] + 1)]
    ax.set_xticklabels(col_label, fontsize=12, family='Times New Roman')

    row_label = ["NAR", "AR_5", "AR_10"]
    ax.set_yticklabels(row_label, fontsize=12, family='Times New Roman', rotation=0)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig('./vis/{}.png'.format(filename), transparent=True, dpi=800)


if __name__ == "__main__":
    config = Options().parse()

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu_index)

    model = get_model(config, device)

    print('loading model from checkpoint: %s' % config.model_path % (config.save_dir_name, config.iter))
    model_cp = pickle.load(open(config.model_path % (config.save_dir_name, config.iter), "rb"))
    model.load_state_dict(model_cp['model_dict'])

    fusion_weight = model.state_dict()['Fusion_module.weight'].cpu().numpy().T
    vis_save_mat(fusion_weight)
    print("save fusion weight  successfully.")
