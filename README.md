## Update
We have added the training and testing code for Human3.6M dataset, which is not reported in our paper. The dataset can
be downloaded from [here](https://pan.baidu.com/s/1b8vnSr8vmRbaJsUMdEsrmg?pwd=zyc2) (original stanford link has crashed, this link is a backup).

Testing results on Human3.6M is:

| Millisecond | 80 | 160 | 320 | 400 | 560 | 1000 |
|-------|-------|-------|-------|-------|-------|-------|
| Average | 10.2 | 22.7 | 47.9 | 59.0 | 77.2 | 111.2 |

## Datasets

[CMU-Mocap](http://mocap.cs.cmu.edu/) is downloaded from repository [Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website. You should put all downloaded datasets into the `./data` directory.

We adopt the data preprocessing from [PGBIG](https://github.com/705062791/PGBIG).

## Training

```
# Human3.6M
python train/train_h36.py --data_dir './data/h3.6m/dataset' --save_dir_name 'h36' --joint_num 22 --S_model_dims 256 --is_mlp_bn true --mlp_dropout 0.5
# CMU-Mocap
python train/train_cmu.py --data_dir './data/cmu' --save_dir_name 'cmu' --joint_num 25 --S_model_dims 1024
# 3DPW
python train/train_3dpw.py --data_dir './data/3DPW/sequenceFiles' --save_dir_name '3dpw' --joint_num 23 --t_pred 30 --t_pred_lst 30,5,10 --S_model_dims 128 --is_mlp_bn true --mlp_dropout 0.7 --num_epoch 60
```

## Testing

```
# Human3.6M
python test/test_h36.py --data_dir './data/h3.6m/dataset' --save_dir_name 'h36' --joint_num 22 --S_model_dims 256 --is_mlp_bn true --mlp_dropout 0.5
# CMU-Mocap
python test/test_cmu.py --data_dir './data/cmu' --save_dir_name 'cmu' --joint_num 25 --S_model_dims 1024
# 3DPW
python test/test_3dpw.py --data_dir './data/3DPW/sequenceFiles' --save_dir_name '3dpw' --joint_num 23 --t_pred 30 --t_pred_lst 30,5,10 --S_model_dims 128 --is_mlp_bn true --mlp_dropout 0.7 --num_epoch 60
```

After running the above commands, you can run utils/get_avg_eval.py to get the averaged test results of the last 10 epochs(the results reported in our paper).
Training&testing logs and model checkpoints can be downloaded from [here](https://pan.baidu.com/s/1WNc5gRKoCjF31vzNgDFOSQ?pwd=055o).

## Visualization
### Predictions
```
# Human3.6M
python vis/draw_pics_h36.py --data_dir './data/h3.6m/dataset' --save_dir_name 'h36' --joint_num 22 --S_model_dims 256 --is_mlp_bn true --mlp_dropout 0.5 --iter 200
# CMU-Mocap
python vis/draw_pics_cmu.py --data_dir './data/cmu' --save_dir_name 'cmu' --joint_num 25 --S_model_dims 1024 --iter 200
# 3DPW
python vis/draw_pics_3dpw.py --data_dir './data/3DPW/sequenceFiles' --save_dir_name '3dpw' --joint_num 23 --t_pred 30 --t_pred_lst 30,5,10 --S_model_dims 128 --is_mlp_bn true --mlp_dropout 0.7 --iter 60
```
