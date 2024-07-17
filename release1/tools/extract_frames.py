import cv2
from os import listdir, makedirs
from os.path import join, exists
from multiprocessing import Pool
import argparse
import warnings

parser = argparse.ArgumentParser("Extract frames.")
parser.add_argument('--video_dir', type=str, required=True,
                    help="The dir containing videos. E.g. 'video_dir'/[01_001.avi, 01_002.avi, ...]")
parser.add_argument('--frame_dir', type=str, required=True,
                    help="The dir to save frames. All the frames of a video will be saved in a directory. E.g. 'frame_dir'/([01_001/, 01_002/, ...])")
parser.add_argument('--frm_name_len', type=int, default=6,
                    help="length of the frame name, e.g., frm_name_len=6: '000000.jpg', '000001.jpg', ...")
parser.add_argument('--skip_first', action='store_true',
                    help="Whether to skip the first frame or not. **For Corridor dataset, please use this option since the first frame is black.**")
parser.add_argument('--workers', type=int, default=48,
                    help="The number of processes.")

args = parser.parse_args()
video_dir: str = args.video_dir
frame_dir: str = args.frame_dir
frm_name_len: int = args.frm_name_len
skip_first: bool = args.skip_first
workers: int = args.workers


def extract_frames(video_name: str):
    print(video_name)

    video_path = join(video_dir, video_name)
    assert exists(video_path), f"{video_path} does not exist!"

    video_cap = cv2.VideoCapture(video_path)
    suc, frame = video_cap.read()

    dst_dir = join(frame_dir, video_name.split('.')[0])
    if not exists(dst_dir):
        makedirs(dst_dir)

    if skip_first:  # skip the first black frame in Corridor dataset
        suc, frame = video_cap.read()

    i_frame = 0
    while suc:
        cv2.imwrite(join(dst_dir, f"{str(i_frame).zfill(frm_name_len)}.jpg"), frame, (cv2.IMWRITE_JPEG_QUALITY, 100))
        i_frame += 1
        suc, frame = video_cap.read()


if __name__ == "__main__":
    warnings.warn("Please use `--skip_first` if the dataset is **Corridor**.")
    video_list = sorted(listdir(video_dir))
    pool = Pool(workers)
    for video_name in video_list:
        pool.apply_async(extract_frames, (video_name,))
    pool.close()
    pool.join()
