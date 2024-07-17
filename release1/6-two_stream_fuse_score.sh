python 6-two_stream_fuse_score.py \
    --CR_sitv 2 --KR_sitv 8 \
    --gtnpz_path "../data.Ave/gt.npz" \
    --score_dict_CR_path "../save.results/release1/score_dict_1231-102911_10.npz" \
    --score_dict_KR_path "../save.results/release1/score_dict_1231-175544_10.npz" \
    --channel 1 $@
# --score_dict_CR_path是指2-test_stream_CR测试得到的score_dict，--score_dict_KR_path是指5-test_stream_KR测试得到的score_dict
# 这里的channel是指选取2-test_stream_CR的score_dict中第几个patch_size的MLE分数
