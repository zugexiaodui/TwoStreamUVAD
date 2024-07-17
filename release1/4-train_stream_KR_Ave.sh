# export CUDA_VISIBLE_DEVICES=3

python 4-train_stream_KR.py \
  --train_data "../save.feats/Ave/release1_I3D_8x8_R50_pre_tmp/Train" \
  --lr 0.003125 --batch_size 256 --iterations 32 --epochs 10 \
  --t_rand_range 150 --lam_m 0.64 \
  --len_hash_code 32 --num_hash_layer 8 --save_freq 1 \
  --print_freq 10 --note "【模板】" $@

# NOTE：训练epoch过多或学习率过大会存在严重“过拟合”问题，导致准确率下降，在各个数据集上都如此。以上超参不是Avenue数据集的最佳训练设置，而且各数据集的训练超参都不太一样，作者由于离校且没有服务器外置硬盘挂载权限，暂时没法拿到之前的训练日志提供最佳训练设置。
