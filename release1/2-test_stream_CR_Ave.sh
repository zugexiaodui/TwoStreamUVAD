export CUDA_VISIBLE_DEVICES="0"

python 2-test_stream_CR.py \
  --data "../data.Ave/frames/Test" --gtnpz_path "../data.Ave/gt.npz" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --encoder_reinit --encoder_rmrelu \
  --resume "../save.ckpts/release1/1-train_stream_CR_**/checkpoint_60.pth.tar" \
  --snippet_len 9 --snippet_itv 2 --nzfill 4 --to_gpu_thres 0 \
  --error_type "MLE" --patch_size 256 128 64 32 16 --patch_stride 8 --lam_l1 0 --crop_fuse_type "max" \
  --workers 1 --note "测试Avenue" $@

