import os 
import os.path as osp
import sys
import cv2
from pathlib import Path
import shutil
import torch
import math
import numpy as np

from tracking_utils.predictor import Predictor
from yolox.utils import fuse_model, get_model_info
from loguru import logger
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from tracking_utils.visualize import plot_tracking

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname( os.path.abspath(__file__) )))) + "/datasets/")
from pretreatment import pretreat
sys.path.append((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
from seg_demo import seg_opengait_image_new
from changejson import writejson

def imageflow_demo(model, save_folder, cfgs, exp, args, gallery=True, oriids=[]):
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)


    if gallery:
        video = cfgs["path"]["gallerypath"]
        # 加一下清除sil和pkl的目录
        # 这个要改一下了，不能清除了，要保存
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
    # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # # folder = "gallery" if gallery else "probe"
    # save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if gallery:
        save_video_name = cfgs["path"]["gallerypath"].split("/")[-1]
    else:
        save_video_name = cfgs["path"]["probepath"].split("/")[-1]
    save_video_path = osp.join(save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
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
        save_video_name = save_video_name.split(".")[0]
        res_file = osp.join(save_folder, f"{save_video_name}.txt")
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

def writeresult(model, save_folder, cfgs, exp, args, pgdict, mintid):
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)

    # pgdict是probe和gallery的id一一对应的一个字典
    video = cfgs["path"]["probepath"]
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(save_folder, exist_ok=True)
    # 写回信息的视频 应该就写回probe的视频就好
    save_video_path = osp.join(save_folder, "result.mp4")
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
        res_file = osp.join(save_folder, "result.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def loadckpt(exp, args, cfgs):
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    ckpt_file = cfgs["model"]["ckpt"]
    logger.info("loading checkpoint")
    print(ckpt_file)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    return model


def seg(exp, args, cfgs, video_save_folder, model):
    # 以后加一个判断video_save_folder是不是空的东西
    if video_save_folder is None:
        pass
    # trt_file = None
    # decoder = None
    # predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    print(cfgs["path"])
    if cfgs["path"]["gallerypath"].endswith(".mp4"):
        _, gids = imageflow_demo(model, video_save_folder, cfgs, exp, args, True)
        # gids 是有gallery的目标的tid
    else:
        print("gallery video is not end with .mp4")
        sys.exit()
    print("################### gids #########################")
    print(gids)
    if cfgs["path"]["probepath"].endswith(".mp4"):
        minid, ids = imageflow_demo(model, video_save_folder, cfgs, exp, args, False, gids.copy())
        # ids是有所有ids的目标！！！！
    else:
        print("probe video is not end with .mp4")

    # pretreat(input_path=Path(cfgs["path"]["savesil_path"]), output_path=Path(cfgs["path"]["pkl_save_path"]), 
    # img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=cfgs["gait"]["dataset"])
    pretreat(Path(cfgs["path"]["savesil_path"]), Path(cfgs["path"]["pkl_save_path"]), 
                args.img_size, args.n_workers, args.verbose, cfgs["gait"]["dataset"])
    
    
    return gids, minid