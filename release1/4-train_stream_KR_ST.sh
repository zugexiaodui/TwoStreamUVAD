export CUDA_VISIBLE_DEVICES=0

python 4-train_stream_KR.py \
  --train_data "../save.feats/ST/..." \
  --lr 0.003125 --batch_size 256 --iterations 32 --epochs 60 \
  --t_rand_range 150 --lam_m 0.64 \
  --len_hash_code 32 --num_hash_layer 8 --save_freq 1 \
  --print_freq 10 --note "【模板】" $@

# --train_data：保存的训练特征文件夹路径