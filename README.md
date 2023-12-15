# AFANAT

Code for paper "Human Motion Prediction via Adaptive Fusing Autoregressive and Non-Autoregressive Attention Networks".

## Datasets

[CMU-Mocap](http://mocap.cs.cmu.edu/) is downloaded from repository [Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website. You should put all downloaded datasets into the `./data` directory.

We adopt the data preprocessing from [PGBIG](https://github.com/705062791/PGBIG).

## Training
For CMU-Mocap:

```
python train/train_cmu.py --data_dir './data/cmu' --joint_num 25 --S_model_dims 1024 --save_dir_name 'cmu'
```

For 3DPW:

```
python train/train_3dpw.py --data_dir './data/3DPW/sequenceFiles' --joint_num 23 --S_model_dims 128 --save_dir_name '3dpw'
```

## Testing
For CMU-Mocap:

```
python test/test_cmu.py --data_dir './data/cmu' --joint_num 25 --S_model_dims 1024 --save_dir_name 'cmu'
```

For 3DPW:

```
python test/test_3dpw.py --data_dir './data/3DPW/sequenceFiles' --joint_num 23 --S_model_dims 128 --save_dir_name '3dpw'
```
After running the above commands, you can run utils/get_avg_eval.py to get the average eval results of last 10 epochs.

## Visualization
### Prediction
```
python vis/draw_pics_cmu.py --data_dir './data/cmu' --joint_num 25 --S_model_dims 1024 --save_dir_name 'cmu' --iter 200
```
### Fusion weight of model
```
python vis/vis_fusion_weight.py --data_dir './data/3DPW/sequenceFiles' --joint_num 23 --S_model_dims 128 --save_dir_name '3dpw' --iter 200
```

## Ablation
Change t_pred_lst argument when running above commands.

