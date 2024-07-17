export CUDA_VISIBLE_DEVICES=0

python 5-test_stream_KR.py \
  --train_data "../save.feats/ST/release1_I3D_8x8_R50_pre_tmp/Train" \
  --test_data "../save.feats/ST/release1_I3D_8x8_R50_pre_tmp/Test" \
  --gtnpz_path "../data.ST/gt.npz" \
  --resume "../save.ckpts/release1/4-train_stream_KR_**/checkpoint_10.pth.tar" \
  --snippet_len 8 --snippet_itv 8 \
  --len_hash_code 32 --num_hash_layer 8 \
  --note "【模板】" $@
