import argparse
import os
import os.path as osp
import torch
import time
import copy

from yolox.exp import get_exp
from loguru import logger

from extractor import extract
from segment import *
from recognition import recognise

cfgs = {  
    "gaitmodel":{
        # "model_type": "Baseline",
        "model_type": "BaselineDemo",
        "cfg_path": "./configs/baseline/baseline_GREW.yaml",
    },
    "path":{
        "jsonpath": "./datasets/CASIA-B/demo.json",
        "gallerypath": "./opengait/demo/output/Inputvideos/demo6.mp4",
        "probepath": "./opengait/demo/output/Inputvideos/demo4.mp4",
        "savesil_path": "./opengait/demo/output/Sil/",
        "pkl_save_path": "./opengait/demo/output/Pkl/",
        "embspath": "./opengait/demo/output/Embs/",
        "whole_pkl_save_path": "./opengait/demo/output/Gaitembs/",
        "video_output_path": "./opengait/demo/output/Outputvideos/"
    },
    "model":{
        "gait_model": "./opengait/demo/checkpoints/gait_model/Baseline-250000.pt",
        "seg_model" : "./opengait/demo/checkpoints/seg_model/human_pp_humansegv1_lite_192x192_inference_model_with_softmax/deploy.yaml",
        "ckpt" :    "./opengait/demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar",
        "exp_file": "./opengait/demo/checkpoints/bytetrack_model/yolox_x_mix_det.py",
    },
    "gait":{
        # "dataset": "CASIAB",
        "dataset": "GREW",
    }
}

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

def main(exp, args, cfgs):
    # 要分成 录入信息 和其他那些东西两部分！！！！
    # 录入信息就是要单独就seg





    output_dir = cfgs["path"]["video_output_path"]
    os.makedirs(output_dir, exist_ok=True)

    # 保存视频，以后要考虑不要保存视频就算特征和比对结果的情况
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # folder = "gallery" if gallery else "probe"
        video_save_folder = osp.join(vis_folder, timestamp)
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    logger.info("Args: {}".format(args))
    model = loadckpt(exp, args, cfgs)
    # print(model)
    # bytetrack的id怎么处理的，，，
    model1 = copy.copy(model)
    # model1 = loadckpt(exp, args, cfgs)

    


    print(output_dir)
    # seg分割图片
    gids, minid = seg(exp, args, cfgs, video_save_folder, cfgs["path"]["gallerypath"], model)
    # gids1, minid1 = seg(exp, args, cfgs, video_save_folder, model1)
    print(gids, minid)
    # print(gids1, minid1)
    # extract提取特征
    # 以后肯定不能有 gids这个参数！！！！
    # gids = [1, 2]
    # minid = 3
    embsdic = extract(cfgs, gids)

    # # recognise 比对环节
    # if args.comparegait:
    #     pgdict = recognise(cfgs, embsdic, gids)

    # if args.save_result:
    #     writeresult(model, video_save_folder, cfgs, exp, args, pgdict, minid)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(cfgs["model"]["exp_file"], None)
    main(exp, args, cfgs)
