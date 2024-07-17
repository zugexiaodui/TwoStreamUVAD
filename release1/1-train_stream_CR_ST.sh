export CUDA_VISIBLE_DEVICES="0"

python 1-train_stream_CR.py \
    --data "../data.ST/training/frames" \
    --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --encoder_rmrelu \
    --lr 0.01 --batch_size 32 --iterations 32 \
    --epochs 60 --schedule 40 \
    --workers 16 --save_freq 10 \
    --snippet_len 9 --snippet_itv 2 --nzfill 4 \
    --print_freq 50 \
    --note "训练ST" --workers 16 $@

 # 这里关键的参数有：
 # CUDA_VISIBLE_DEVICES：控制使用哪些显卡，这里建议只使用1个显卡训练，作者发现分散到多个显卡上训练容易使准确率下降，猜测可能是BatchNorm引起的。
 # --data：保存训练视频帧的目录，该目录下每个视频为一个文件夹，文件夹下是视频帧的图片，比如01_001/{0000.jpg, 0001.jpg, ...}。从视频提取帧的代码本仓库不提供，需要自己写一下，建议使用opencv提取，保存图片时imwrite函数设置图片质量为100%，直接使用ffmpeg可能会导致抽取的帧数不一致。[X]可在tools文件夹下找到extract_frames.py用于把视频转化为帧。mvfile_corridor.py用于把原Corridor数据集的视频去除多余的一级文件夹。
 # --encoder_path：这里从pth文件直接加载模型，该pth文件是作者自己从pyslowfast(https://github.com/facebookresearch/SlowFast)库提供的Kinetics-400预训练模型转换的，已提供在data.model文件夹下，需要自己修改路径。
 # --nzfill：这里4是因为每个视频帧的格式为"XXXX.jpg"，根据需要修改。后续有的程序没有在命令行指定nzfill参数，如果遇到文件读取错误，请在.py文件中对应位置修改zfill()参数。
 # 其他参数含义请结合程序查看。
 