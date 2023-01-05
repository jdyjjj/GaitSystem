import argparse
import os
import os.path as osp
import pickle
import time
import cv2
import torch
import pandas as pd
from torch import nn
import torch
import math
import numpy as np
from loguru import logger
import sys
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
# from yolox.tracking_utils.timer import Timer
from timer import Timer
from pathlib import Path
# from yolox.utils.visualize import plot_tracking
from visualize import plot_tracking
# from yolox.tracker.byte_tracker import BYTETracker
from tracker.byte_tracker import BYTETracker
import shutil
root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
# print(root)
sys.path.append(root)
# config = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/"
sys.path.append((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
# print((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
from seg_demo import seg_opengait_image_new
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))) + "/datasets/")
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))) + "/datasets/")
from pretreatment import pretreat
from utils import config_loader
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/models/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
print(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from modeling import models
import gait_compare as gc
from changejson import writejson, getid, getidlist
from data_augment import preproc

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# 运行指令：
# python3 opengait/demo/tools/track_opengait_demo.py video --fp16 --fuse --save_result  --phase test --compare
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
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
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
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
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


def loadModel(model_type, cfg_path, compare=False):
    Model = getattr(models, model_type)
    # with open(cfg_path, 'r') as stream:
    #     cfgs = yaml.safe_load(stream)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    model.compare = compare
    return model, cfgs

cfgs = {  
"gaitmodel":{
    "model_type": "Baseline",
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

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def imageflow_demo(predictor, vis_folder, current_time, cfgs, args, gallery=True, oriids=[]):
    if gallery:
        video = cfgs["path"]["gallerypath"]
        #加一下清除sil和pkl的目录
        if os.path.exists(cfgs["path"]["savesil_path"]):
            shutil.rmtree(cfgs["path"]["savesil_path"])
        if os.path.exists(cfgs["path"]["pkl_save_path"]):
            shutil.rmtree(cfgs["path"]["pkl_save_path"])
    else:
        video = cfgs["path"]["probepath"]
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    if gallery:
        gids = []
    else:
        pids = []
    ids = oriids
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # folder = "gallery" if gallery else "probe"
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if gallery:
        save_video_path = osp.join(save_folder, cfgs["path"]["gallerypath"].split("/")[-1])
    else:
        save_video_path = osp.join(save_folder, cfgs["path"]["probepath"].split("/")[-1])
    print(f"###video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    mark = False
    mintid = 0
    results = []
    while True:
        # if frame_id == 90:
        #     break
        print("frame {} begins".format(frame_id))
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if not gallery and not mark:
                        mark = True
                        mintid = tid
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # track 记录的信息 记录一下 方便之后用gait判断之后写回到视频里面
                        print(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1")

                        # 清除一下目录, 并且把第一次出现的tid加入ids里面
                        tidstr = "{:03d}".format(tid)
                        if gallery:
                            tidstr = "{}{:03d}".format("g",tid)
                            savesil_path = osp.join(cfgs["path"]["savesil_path"], tidstr, "undefined", "undefined")
                            gids.append(tid)
                        else:
                            tidstr = "{}{:03d}".format("p",tid)
                            savesil_path = osp.join(cfgs["path"]["savesil_path"], tidstr, "undefined", "undefined")
                            pids.append(tid)
                        ids.append(tid)

                        x = tlwh[0]
                        y = tlwh[1]
                        width = tlwh[2]
                        height = tlwh[3]

                        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                        w, h = x2 - x1, y2 - y1
                        x1_new = max(0, int(x1 - 0.1 * w))
                        x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                        y1_new = max(0, int(y1 - 0.1 * h))
                        y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(y2 + 0.1 * h))
                        
                        new_w = x2_new - x1_new
                        new_h = y2_new - y1_new
                        tmp = frame[y1_new: y2_new, x1_new: x2_new, :]

                        # 先用tid来进行命名
                        save_name = "{:03d}-{:03d}.png".format(tid, frame_id)
                        
                        #居中调整seg
                        side = max(new_w,new_h)
                        tmp_new = [[[255,255,255]]*side]*side
                        tmp_new = np.array(tmp_new)
                        width01 = math.floor((side-new_w)/2)
                        height01 = math.floor((side-new_h)/2)
                        tmp_new[int(height01):int(height01+new_h),int(width01):int(width01+new_w),:] = tmp
                        tmp_new=tmp_new.astype(np.uint8)
                        tmp = cv2.resize(tmp_new,(192,192))
                        #居中调整seg
                        seg_opengait_image_new(tmp, cfgs["model"]["seg_model"], save_name, savesil_path)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    # ids去重, 以后再加个升序排列
    ids=list(set(ids))
    if gallery:
        gids=list(set(gids))
        return mintid, gids
    else:
        pids=list(set(pids))

    # 在处理完probe后再，写入要求的json文件里面，都写成testset
    if not gallery:
        writejson(cfgs["path"]["jsonpath"], ids, pids)
    return mintid, ids

def writeresult(predictor, vis_folder, current_time, cfgs, args, pgdict, mintid):
    # pgdict是probe和gallery的id一一对应的一个字典
    video = cfgs["path"]["probepath"]
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    # 写回信息的视频 应该就写回probe的视频就好
    save_video_path = osp.join(save_folder, "result")
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    # 这里mark和diff的思路不知道写的对不对
    mark = False
    diff = 0
    while True:
        # if frame_id == 90:
        #     break
        if frame_id % 40 == 0:
            print("frame {} begins".format(frame_id))
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    if not mark:
                        mark = True
                        diff = t.track_id - mintid
                    # print(t.track_id,diff)
                    tid = pgdict[t.track_id - diff]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def gait(cfgs, embsdict:dict, gids):
    print("========= Begin storing gait information ==========")
    gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    print("######## gaitmodel.__class__ #########")
    print(gaitmodel.__class__)
    cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], train=False)
    if os.path.exists(cfgs["path"]["whole_pkl_save_path"]):
        shutil.rmtree(cfgs["path"]["whole_pkl_save_path"])
    for inputs in loader:
        ipts = gaitmodel.inputs_pretreament(inputs)
        id = inputs[1][0] + 1
        type = inputs[2][0] 
        view = inputs[3][0]
        print("########### gait id #################")
        print(id)
        print(gids)
        if id in gids:
            savePklPath = "{}/{}{:03d}/{}/{}".format(cfgs["path"]["whole_pkl_save_path"], "g", id, type, view)
            id = "g" + str(id)
        else:
            savePklPath = "{}/{}{:03d}/{}/{}".format(cfgs["path"]["whole_pkl_save_path"], "p", id, type, view)
            id = "p" + str(id)
        print(savePklPath)
        if not os.path.exists(savePklPath):
            os.makedirs(savePklPath)
        savePklName = "{}/{}.pkl".format(savePklPath, inputs[3][0])
        retval, embs = gaitmodel.forward(ipts)
        pkl = open(savePklName, 'wb')
        pickle.dump(embs, pkl)
        if id not in embsdict:
            embsdict[id] = []
        ap = {}
        ap[type] = {}
        ap[type][view] = embs
        
        if len(embsdict[id]) == 0:
            embsdict[id].append(ap)
        else:
            for idx, e in enumerate(embsdict[id]):
                if str(e) == str(ap):
                    break
                elif idx == len(embsdict[id])-1:
                    embsdict[id].append(ap)            
        del ipts

def gaitcompare(cfgs, embsdict:dict, gids):
    print("========= Begin comparing..... ==========")
    gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], train=False)
    pg_dict = {}
    matrix_dict = {}
    for inputs in loader:
        # pid gid 要剔除一下pid
        # 遇到gid要跳过
        pid = inputs[1][0]+1
        if pid in gids:
            # 那说明是gid，跳过
            continue
        realpid = getid(cfgs["path"]["jsonpath"], pid)
        ipts = gaitmodel.inputs_pretreament(inputs)
        retval, embs = gaitmodel.forward(ipts)
        # print("################################# ids ##########################")
        # print(realpid, pid)
        # retval是传入pid的信息
        gid, iddict = gc.compareid(retval,embsdict,cfgs["path"]["jsonpath"], 100)
        # matrix_dict[realprobeid] = iddict
        # print(embsdict)
        # print("################################# galleryid")
        # # 不过现在找到的是他自己是pid,按理来说就替换g就好
        # print(gid)
        gid = gid.replace("g","")
        # gid = gid.replace("p","")
        print(gid)
        realgid = getid(cfgs["path"]["jsonpath"], int(gid))
        # print(realgid, gid, realpid, pid)

        # print("###### realgid #######")
        # print(realgid)
        realgid = realgid.replace("g","")
        # realgid = realgid.replace("p","")
        stridlist = list(realgid)
        temp = ''.join(stridlist)
        galleryidnum = int(temp)

        # realpid = realpid.replace("g","")
        # print("###### realpid #######")
        # print(realpid)
        realpid = realpid.replace("p","")
        # print(stridlist)
        stridlist = list(realpid)
        temp = ''.join(stridlist)
        probeidnum = int(temp)
        pg_dict[probeidnum] = galleryidnum
        matrix_dict[probeidnum] = iddict
    print("##################matrix_dict")
    print(matrix_dict)
    return pg_dict

def main(exp, args):
    if 1:
        output_dir = cfgs["path"]["video_output_path"]
        os.makedirs(output_dir, exist_ok=True)

        if args.save_result:
            vis_folder = osp.join(output_dir, "track_vis")
            os.makedirs(vis_folder, exist_ok=True)
        print(output_dir)
        # "video_output_path"

        if args.trt:
            args.device = "gpu"
        args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

        logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model().to(args.device)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if not args.trt:
            if cfgs["model"]["ckpt"] is None:
                ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
            else:
                ckpt_file = cfgs["model"]["ckpt"]
            logger.info("loading checkpoint")
            print(ckpt_file)
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # print(ckpt)
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()  # to FP16

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
    if 1:
        predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
        # del model
        current_time = time.localtime()
        print(cfgs["path"])
        # 视频输入不能是三级目录了就是一个mp4的video, 目前是gallery部分
        # 把probe和gallery要融合起来！！！然后就一次pretreat！！！
        if cfgs["path"]["gallerypath"].endswith(".mp4"):
            _, gids = imageflow_demo(predictor, vis_folder, current_time, cfgs, args, True)
            # gids 是有gallery的目标的tid
        else:
            print("gallery video is not end with .mp4")
            sys.exit()
        print("################### gids #########################")
        print(gids)
        if cfgs["path"]["probepath"].endswith(".mp4"):
            mintid, ids = imageflow_demo(predictor, vis_folder, current_time, cfgs, args, False, gids.copy())
            # ids是有所有ids的目标！！！！
        else:
            print("probe video is not end with .mp4")
        #     sys.exit()
        # print("################### ids #########################")
        # print(mintid)
        # print(ids)
        # print(gids)
        # # path要改，前面加个gallery的
        # deletedirs(cfgs, True)
        pretreat(input_path=Path(cfgs["path"]["savesil_path"]), output_path=Path(cfgs["path"]["pkl_save_path"]), img_size=args.img_size, 
                                    workers=args.n_workers, verbose=args.verbose, dataset=cfgs["gait"]["dataset"])

    # 加载embs, 先固定名字是embeddings吧
    # load embslist
    # embslist是记录特征矩阵的dic,id:[]

    embsdic = {}
    if not osp.exists(cfgs["path"]["embspath"]):
        os.makedirs(cfgs["path"]["embspath"])
    embs_path = "{}{}.pkl".format(cfgs["path"]["embspath"], "embeddings")
    # gids = [1,2]
    # gait
    gait(cfgs, embsdic, gids)
    print("####### embs #########")
    print(embsdic)
    # save embslist
    pkl = open(embs_path, 'wb')
    pickle.dump(embsdic, pkl)

    # # # compare
    if args.comparegait:
        pgdict = gaitcompare(cfgs, embsdic, gids)
        print("################## probe - gallery ##################")
        print(pgdict)
        if args.save_result:
            writeresult(predictor, vis_folder, current_time, cfgs, args, pgdict, mintid)
    
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(cfgs["model"]["exp_file"], None)
    # exp = get_exp(cfgs["model"]["exp_file"], cfgs["model"]["exp_file"])
    main(exp, args)