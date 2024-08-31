# export CUDA_VISIBLE_DEVICES=3

python 4-train_stream_KR.py \
  --train_data "../save.feats/Ave/release1_I3D_8x8_R50_pre_tmp/Train" \
  --lr 0.003125 --batch_size 256 --iterations 32 --epochs 10 \
  --t_rand_range 150 --lam_m 0.64 \
  --len_hash_code 32 --num_hash_layer 8 --save_freq 1 \
  --print_freq 10 --note "【模板】" $@

# NOTE：训练epoch过多或学习率过大会存在严重“过拟合”问题，导致准确率下降，在各个数据集上都如此。请参考论文设置。
