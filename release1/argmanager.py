import argparse


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _stream_CR_base_parser(training: bool):
    parser = argparse.ArgumentParser(description=f'')

    # Path settings
    parser.add_argument('--data', type=str, required=training,
                        help='Path to the data.')
    parser.add_argument('--resume', default="", type=str,
                        help='')

    # Resource usage settings
    parser.add_argument('--workers', default=6 if training else 1, type=int,
                        help='Number of data loading workers')

    # Model settings
    parser.add_argument('--encoder_path', default="", type=str,
                        help='')
    parser.add_argument('--encoder_reinit', action="store_true",
                        help='')
    parser.add_argument('--encoder_rmrelu', action="store_true",
                        help='')
    parser.add_argument('--only_feat', action="store_true",
                        help='')

    # Dataset settings
    parser.add_argument('--snippet_len', type=int, choices=(8, 9),
                        help="")
    parser.add_argument('--snippet_itv', type=int, default=2,
                        help="")
    parser.add_argument('--nzfill', default=4, type=int,
                        help="XXXX.jpg")

    # Other settings
    parser.add_argument('--log_root', default="save.logs", type=str,
                        help='')
    parser.add_argument('--note', default="", type=str,
                        help='A note for this experiment')
    parser.add_argument('--print_model', action="store_true",
                        help='')
    parser.add_argument('--debug_mode', action="store_true",
                        help='')

    return parser


def train_stream_CR_parser():
    parser = _stream_CR_base_parser(training=True)

    # Optimizer settings
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        help='Learning rate.')
    parser.add_argument('--fr', '--funetune_rate', default=0, type=float,
                        help='')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', default=60, type=int,
                        help='Number of total epochs to run.')
    parser.add_argument('--schedule', default=[40], nargs='*', type=int,
                        help='Learning rate schedule.')

    # Dataset settings
    parser.add_argument('--iterations', default=32, type=int,
                        help='A way to simulate more epochs.')
    parser.add_argument('--fixed_3crop', action="store_true",
                        help='For Corridor')

    # Loss weight settings
    parser.add_argument('--lam_l1', default=1.0, type=float,
                        help="")

    # Saving and logging settings
    parser.add_argument('--ckpt_root', default="save.ckpts", type=str,
                        help='')
    parser.add_argument('--tbxs_root', default="save.tbxs", type=str,
                        help='')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='Save frequency.')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency.')

    return parser


def test_stream_CR_parser():
    parser = _stream_CR_base_parser(training=False)

    # Path settings
    parser.add_argument('--gtnpz_path', type=str, required=True,
                        help='Path to groundtruth npz file.')
    parser.add_argument('--score_dict_path', type=str, default="",
                        help='Only calculate AUCs for this score_dict. --data and --resume will be ignored.')
    parser.add_argument('--vid_dir', type=str, default="",
                        help='Use the .avi videos instead of the frame in .jpg')

    # Dataset settings
    parser.add_argument('--to_gpu_thres', type=int, default=0,
                        help="=0: all cpu, -1: all gpu, >0: value gpu")

    # Resource usage settings
    parser.add_argument('--threads', default=24, type=int,
                        help='Number of threads used by pytorch')

    # Error settings
    parser.add_argument('--lam_l1', default=1.0, type=float,
                        help="")
    parser.add_argument('--error_type', type=str, default='MLE', choices=('FLE', 'MLE'),
                        help='')
    parser.add_argument('--patch_size', type=int, nargs='+',
                        help='')
    parser.add_argument('--patch_stride', type=int, default=8,
                        help='')
    parser.add_argument('--use_channel_l2', action="store_true",
                        help='')
    parser.add_argument('--crop_fuse_type', type=str, default='mean', choices=('mean', 'max'),
                        help='')

    # Saving settings
    parser.add_argument('--result_root', default="save.results", type=str,
                        help='')
    return parser


def feat_stream_KR_parser():
    parser = _stream_CR_base_parser(training=False)

    # Dataset settings
    parser.add_argument('--to_gpu_thres', type=int, default=0,
                        help="=0: all cpu, -1: all gpu, >0: value gpu")

    # Resource usage settings
    parser.add_argument('--threads', default=24, type=int,
                        help='Number of threads used by pytorch')

    # Saveing and restoring settings:
    parser.add_argument('--feat_root', type=str, default="save.feats",
                        help='')
    parser.add_argument('--save_suffix', type=str, default="",
                        help='')
    parser.add_argument('--last_finished', type=str, default="",
                        help='')

    return parser


def _stream_KR_base_parser(training: bool):
    parser = argparse.ArgumentParser(description=f'')

    # Path settings
    if training:
        parser.add_argument('--train_data', type=str, required=True,
                            help='Path to the data.')
    else:
        parser.add_argument('--train_data', type=str,
                            help='Path to the training features.')
        parser.add_argument('--test_data', type=str,
                            help='Path to the testing features.')

    # Snippet settings
    parser.add_argument('--snippet_len', type=int, choices=(8, 9), default=8,
                        help="")
    parser.add_argument('--snippet_itv', type=int, default=8,
                        help="")

    # Resource usage settings
    parser.add_argument('--workers', default=16 if training else 4, type=int,
                        help='Number of data loading workers')

    # Model settingts
    parser.add_argument('--len_hash_code', default=32, type=int,
                        help='Length of hash codes, i.e., r')
    parser.add_argument('--num_hash_layer', default=8, type=int,
                        help='Number of hash layers, i.e., b')

    # Other settings
    parser.add_argument('--log_root', default="save.logs", type=str,
                        help='')
    parser.add_argument('--note', default="", type=str,
                        help='A note for this experiment')
    parser.add_argument('--print_model', action="store_true",
                        help='')
    parser.add_argument('--debug_mode', action="store_true",
                        help='')

    return parser


def train_stream_KR_parser():
    parser = _stream_KR_base_parser(training=True)

    # Common training settings
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of total epochs to run.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size.')

    # Dataset settings:
    parser.add_argument('--iterations', default=1, type=int,
                        help='A way to simulate more epochs.')
    parser.add_argument('--t_rand_range', default=150, type=int,
                        help="Sampling offset, i.e., deltaVar_t")

    # Loss settings
    parser.add_argument('--lam_m', default=0., type=float,
                        help='')

    # Optimizer settings
    parser.add_argument('--momentum', default=0.9, type=float, choices=(0.9,),
                        help='Momentum of SGD pptimizer.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, choices=(1e-4,),
                        help='Weight decay.')
    parser.add_argument('--schedule', default=[999], nargs='*', type=int, choices=(999,),
                        help='Learning rate schedule.')

    # Saving and logging settings
    parser.add_argument('--ckpt_root', default="save.ckpts", type=str,
                        help='')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Save frequency.')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency.')
    return parser


def test_stream_KR_parser():
    parser = _stream_KR_base_parser(training=False)

    # Path settings
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to a checkpoint')
    parser.add_argument('--gtnpz_path', type=str, required=True,
                        help='Path to groundtruth npz file.')
    parser.add_argument('--il2sh_inst_path', type=str, default="",
                        help='Use this iL2SH instance in query stage. --index_data and --resume will be ignored.')
    parser.add_argument('--score_dict_path', type=str, default="",
                        help='Only calculate AUCs for this score_dict. ' +
                        '--index_data, --query_data, --resume and --il2sh_inst_path will be ignored.')

    # Resource usage settings
    parser.add_argument('--threads', default=24, type=int,
                        help='Number of threads used by pytorch')

    # Saving settings
    parser.add_argument('--result_root', default="save.results", type=str,
                        help='')

    return parser


def two_stream_fuse_score_parser():
    parser = argparse.ArgumentParser(description=f'')

    # Path settings
    parser.add_argument('--gtnpz_path', type=str, required=True,
                        help='Path to groundtruth npz file.')
    parser.add_argument('--score_dict_CR_path', type=str, required=True,
                        help='Path to context reconstrction score dict.')
    parser.add_argument('--score_dict_KR_path', type=str, required=True,
                        help='Path to knowledge retrieval score dict.')

    # Dataset Settings
    parser.add_argument('--CR_slen', type=int, default=9,
                        help="")
    parser.add_argument('--CR_sitv', type=int, default=2,
                        help="")
    parser.add_argument('--KR_slen', type=int, default=8,
                        help="")
    parser.add_argument('--KR_sitv', type=int, default=8,
                        help="")

    parser.add_argument('--CR_weight', type=float, default=1.,
                        help="")
    parser.add_argument('--KR_weight', type=float, default=1.,
                        help="")

    # Score channel settings
    parser.add_argument('--channel', type=int, default=0,
                        help='')

    # Saving settings
    parser.add_argument('--result_root', default="save.results", type=str,
                        help='')

    # Other settings
    parser.add_argument('--log_root', default="save.logs", type=str,
                        help='')
    parser.add_argument('--note', default="", type=str,
                        help='A note for this experiment')
    parser.add_argument('--debug_mode', action="store_true",
                        help='')
    return parser


if __name__ == '__main__':
    pass
