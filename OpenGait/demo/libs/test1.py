import argparse
import os
import os.path as osp
import torch
import time
import copy
import pickle

from yolox.exp import get_exp
from loguru import logger

from extractor import extract
from segment import *
from recognition import recognise

def make_parser():
    parser = argparse.ArgumentParser("OpenGait Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument('-nw', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    parser.add_argument("--img_size",type=int, default=64, help="img_size")
    parser.add_argument("--workers",type=int, default=4, help="workers")
    parser.add_argument("--verbose",type=bool, default=False, help="verbose")
    # gait
    parser.add_argument('--local_rank', type=int, default=0,
                        help="passed by torch.distributed.launch module")
    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help="choose train or test phase")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--iter', default=0, help="iter to restore")
    # compare
    parser.add_argument(
        "--comparegait",
        dest="comparegait",
        default=False,
        action="store_true",
        help="whether to compare",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    return parser

def main(args):
    output_dir = "./opengait/demo/output/Outputvideos/"
    os.makedirs(output_dir, exist_ok=True)
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        video_save_folder = osp.join(vis_folder, timestamp)

    # seg分割图片
    probe = "./opengait/demo/output/Inputvideos/demo4.mp4"
    gallery = "./opengait/demo/output/Inputvideos/demo6.mp4"
    video = "./opengait/demo/output/Inputvideos/demo1.mp4"
    seg(args, video_save_folder, probe)
    seg(args, video_save_folder, gallery)
    seg(args, video_save_folder, video)

    # extract提取特征
    extract(probe)
    extract(gallery)
    extract(video)

    # # recognise 比对环节
    embspath = "./opengait/demo/output/Embs/"
    if args.comparegait:
        embs_path = "{}{}.pkl".format(embspath, "embeddings")
        if osp.exists(embs_path):
            print("========= Load Embs..... ==========")
            with open(embs_path,'rb') as f:	
                embsdic = pickle.load(f)	
                print("========= Finish Load Embs..... ==========")
        pgdict1 = recognise(embsdic, probe)
        pgdict2 = recognise(embsdic, video)
        pgdict3 = recognise(embsdic, gallery)

        if args.save_result:
            writeresult(video_save_folder, args, pgdict1, probe)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
