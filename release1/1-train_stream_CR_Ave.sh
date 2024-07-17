export CUDA_VISIBLE_DEVICES="0"

python 1-train_stream_CR.py \
    --data "../data.Ave/frames/Train" \
    --encoder_path "../save.pretrainings/I3D_8x8_R50_pre.pth" --encoder_rmrelu \
    --lr 0.01 --batch_size 32 --iterations 320 \
    --epochs 60 --schedule 40 \
    --workers 16 --save_freq 10 \
    --snippet_len 9 --snippet_itv 2 --nzfill 4 \
    --print_freq 50 \
    --note "训练Avenue" --workers 16 $@
