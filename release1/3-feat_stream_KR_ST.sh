export CUDA_VISIBLE_DEVICES="3"

python 3-feat_stream_KR.py \
  --data "../data.ST/training/frames" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --only_feat \
  --snippet_len 8 --snippet_itv 8 --nzfill 4 --to_gpu_thres 0 \
  --workers 1 --note "【模板】" --save_suffix "tmp" $@

python 3-feat_stream_KR.py \
  --data "../data.ST/testing/frames" \
  --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --only_feat \
  --snippet_len 8 --snippet_itv 8 --nzfill 4 --to_gpu_thres 0 \
  --workers 1 --note "【模板】" --save_suffix "tmp" $@

# 这是用于提取KR流特征的脚本，各参数与2-test~.py是一样的含义，提取的特征会保存到上一层目录(即与release1文件夹同级目录)的save.feats文件夹下，确保硬盘空间足够。这里也可以把workers设大一些加快速度。
# --save_suffix：这个参数是特征文件夹的后缀，可根据需要修改或去掉
# 为了保证之后训练和测试时的数据读取效率，这里提取特征时保存的训练特征目录格式与测试特征目录格式不一样，当前是训练数据还是测试数据是根据路径里是否含'Train'确定的，可查看py程序，因此要确保能够区分训练和测试数据。
