import torch
import torch.nn as nn
from einops import rearrange
import models.S_TransformerEncoder as S_Trans_Encoder
import models.PositionEncodings as PositionEncodings


class Fusion_module(nn.Module):
    def __init__(self, joint_num, t_pred, t_pred_lst, is_norm, device):
        super(Fusion_module, self).__init__()
        self.joint_num = joint_num
        self.t_pred = t_pred
        self.t_pred_lst = t_pred_lst
        self.n = len(t_pred_lst)
        self.is_norm = is_norm
        self.weight = nn.Parameter(
            torch.tensor([[1.0 / self.n for j in range(self.n)] for i in range(t_pred)], device=device))

    def forward(self, out_res_lst):
        out_res_ = []
        for i in range(self.t_pred):
            s = torch.zeros_like(out_res_lst[0][0])
            w = 0
            for k in range(self.n):
                s = s + out_res_lst[k][i] * self.weight[i][k]
                w = w + self.weight[i][k]
            if self.is_norm:
                out_res_.append(s / w)
            else:
                out_res_.append(s)
        out_res_ = torch.stack(out_res_, dim=0)
        return out_res_


class AFANAT(nn.Module):
    def __init__(
            self,
            T_enc_hiddims,
            S_trans_enc_num_layers,
            S_model_dims,
            S_num_heads,
            S_dim_feedforward,
            S_dropout_rate,
            T_dec_hiddims,
            t_his,
            t_pred,
            t_pred_lst,
            joint_num,
            batch_size,
            is_norm,
            device,
            posit_encoding_params=(10000, 1)
    ):
        super(AFANAT, self).__init__()
        self.t_his = t_his
        self.t_pred = t_pred
        self.joint_num = joint_num
        self.batch_size = batch_size
        self.device = device
        self.t_pred_lst = t_pred_lst

        self.posit_encoder = PositionEncodings.PositionEncodings1D(
            num_pos_feats=S_model_dims,
            temperature=posit_encoding_params[0],
            alpha=posit_encoding_params[1]
        )

        self.T_encoder_lst = nn.ModuleList([])
        for t_pred_ in t_pred_lst:
            self.T_encoder_lst.append(
                nn.Sequential(
                    nn.Linear(t_his * 3, T_enc_hiddims),
                    nn.Tanh(),
                    nn.Linear(T_enc_hiddims, S_model_dims),
                )
            )

        self.S_trans_encoder_lst = nn.ModuleList([])
        for t_pred_ in t_pred_lst:
            self.S_trans_encoder_lst.append(
                S_Trans_Encoder.S_TransformerEncoder(
                    num_layers=S_trans_enc_num_layers,
                    model_dims=S_model_dims,
                    num_heads=S_num_heads,
                    dim_feedforward=S_dim_feedforward,
                    dropout_rate=S_dropout_rate,
                    joint_num=joint_num
                )
            )

        self.T_decoder_lst = nn.ModuleList([])
        for t_pred_ in t_pred_lst:
            self.T_decoder_lst.append(
                nn.Sequential(
                    nn.Linear(S_model_dims, T_dec_hiddims),
                    nn.Tanh(),
                    nn.Linear(T_dec_hiddims, t_pred_ * 3)
                )
            )

        if len(self.t_pred_lst) > 1:
            self.Fusion_module = Fusion_module(joint_num=joint_num, t_pred=t_pred, t_pred_lst=t_pred_lst,
                                               is_norm=is_norm, device=device)

    def forward(self, x):
        S_posit_encoding = self.posit_encoder(self.joint_num).type(torch.float64).to(self.device)
        out_res_lst = []
        for k, t_pred_ in enumerate(self.t_pred_lst):
            out_res = []
            input_x = x
            for i in range(int(self.t_pred / t_pred_ + 0.5)):
                output = self.T_encoder_lst[k](rearrange(input_x, 'b t c d -> b c (t d)'))

                output, att_weight = self.S_trans_encoder_lst[k](
                    output,
                    S_posit_encoding
                )

                stacked_last_frame = torch.stack([input_x[:, -1] for i in range(t_pred_)], dim=1)
                stacked_last_frame = rearrange(stacked_last_frame, 'b t c d -> b c (t d)')

                output = self.T_decoder_lst[k](output) + stacked_last_frame
                output = rearrange(output, 'b c (t d) -> b t c d', t=t_pred_)

                for i in range(t_pred_):
                    out_res.append(output[:, i])

                if t_pred_ < self.t_pred:
                    if self.t_his - t_pred_ > 0:
                        input_x = torch.cat([input_x[:, -(self.t_his - t_pred_):], output], dim=1)
                    else:
                        input_x = output[:, -self.t_his:]

            out_res = torch.stack(out_res, dim=0)[:self.t_pred]
            out_res_lst.append(out_res)

        if len(self.t_pred_lst) > 1:
            out_res = self.Fusion_module(out_res_lst)

        return out_res_lst, out_res


def get_model(config, device):
    model = AFANAT(
        T_enc_hiddims=config.T_enc_hiddims,
        S_trans_enc_num_layers=config.S_trans_enc_num_layers,
        S_model_dims=config.S_model_dims,
        S_num_heads=config.S_num_heads,
        S_dim_feedforward=config.S_dim_feedforward,
        S_dropout_rate=config.S_dropout_rate,
        T_dec_hiddims=config.T_dec_hiddims,
        t_his=config.t_his,
        t_pred=config.t_pred,
        t_pred_lst=config.t_pred_lst,
        joint_num=config.joint_num,
        batch_size=config.batch_size,
        is_norm=config.is_norm,
        device=device,
    )
    return model
