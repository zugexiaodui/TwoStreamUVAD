# export CUDA_VISIBLE_DEVICES="3"

python 2-test_stream_CR.py \
  --data "../data.ST/testing/frames" --gtnpz_path "../data.ST/gt.npz" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --encoder_reinit --encoder_rmrelu \
  --resume "../save.ckpts/release1/1-train_stream_CR_**/checkpoint_60.pth.tar" \
  --snippet_len 9 --snippet_itv 2 --nzfill 3 --to_gpu_thres 0 \
  --error_type "MLE" --patch_size 256 128 64 32 16 --patch_stride 8 --use_channel_l2 --lam_l1 1.0 --crop_fuse_type "mean" \
  --workers 1 --note "测试ST" $@

# --data：测试视频帧文件夹，与训练数据的存放格式相同
# --gtnpz_path：groundtruth文件，以npz格式保存，此处提供了
# --resume：要加载的训练好的模型
# --workers：使用几个进程测试，会把进程分布到CUDA_VISIBLE_DEVICES上，建议是CUDA_VISIBLE_DEVICES的倍数
# --to_gpu_thres：Dataset加载数据时，当一个视频的帧数不超过这个阈值就会把这些帧提前放在GPU上，因此会占用额外的显存，如果超过阈值就还是在CPU上