export CUDA_VISIBLE_DEVICES="3"

python 3-feat_stream_KR.py \
  --data "../data.Ave/frames/Train" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --only_feat \
  --snippet_len 8 --snippet_itv 8 --nzfill 4 --to_gpu_thres 0 \
  --workers 1 --note "【模板】" --save_suffix "tmp" $@

python 3-feat_stream_KR.py \
  --data "../data.Ave/frames/Test" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --only_feat \
  --snippet_len 8 --snippet_itv 8 --nzfill 4 --to_gpu_thres 0 \
  --workers 1 --note "【模板】" --save_suffix "tmp" $@
