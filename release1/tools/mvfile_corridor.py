from os import system, listdir
from os.path import join
import argparse

parser = argparse.ArgumentParser("Move all the videos (.avi) in one directory for Corridor dataset.")
parser.add_argument("--video_dir", type=str, required=True,
                    help="The original Train/Test directory of Corridor dataset. E.g. 'video_dir'/[000001/000001.avi, 000002/000002.avi, ...]" +
                    " ==> 'video_dir'/[001.avi, 002.avi, ...]")

args = parser.parse_args()
vid_dir: str = args.video_dir

for _vid_name in sorted(listdir(vid_dir)):
    src_path = join(vid_dir, _vid_name, _vid_name + '.avi')
    dst_path = join(vid_dir, _vid_name[3:] + '.avi')  # del the first three 0s, e.g. '000001/000001.avi' ==> '001.avi'

    system(f"mv {src_path} {dst_path}")
    system(f"rmdir {join(vid_dir, _vid_name)}")
