import argparse
import os
import os.path as osp
import torch
import time
import copy
import pickle

from yolox.exp import get_exp
from loguru import logger

from extractor import *
from segment import *
from recognition import *

def main():
    output_dir = "./demo/output/Outputvideos/"
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(vis_folder, timestamp)
    
    save_root = './demo/output/'
    # seg分割图片
    probe_video_path = "./demo/output/Inputvideos/demo4.mp4"
    gallery_video_path = "./demo/output/Inputvideos/demo6.mp4"
    # video = "./demo/output/Inputvideos/demo1.mp4"
    # probe_sil = seg(probe_video_path, save_root+'/silhouette')
    # seg(save_root+'/segmentation', probe)
    # gallery_sil = seg(gallery_video_path, video_save_folder)
    # print(gallery_sil)
    # seg(video_save_folder, video)

    # extract提取特征
    # probe_feat = extract(probe_sil, save_root+'/Embs')
    # probe_feat = extract_sil(probe_sil, save_root+'/Embs')
    # # extract(save_root+'/embeddings', probe)
    probe_sil = getsil(probe_video_path)
    gallery_sil = getsil(gallery_video_path)
    # extract(gallery_video_path)
    probe_feat = extract_sil(probe_sil, save_root+'//')
    gallery_feat = extract_sil(gallery_sil, save_root+'//')
    # extract(video)

    # # # recognise 比对环节
    # embspath = "./demo/output/Embs/"
    # embs_path = "{}{}.pkl".format(embspath, "embeddings")
    # if osp.exists(embs_path):
    #     print("========= Load Embs..... ==========")
    #     with open(embs_path,'rb') as f:	
    #         embsdic = pickle.load(f)	
    #         # print(embsdic)
    #         print("========= Finish Load Embs..... ==========")
    # pgdict = recognise_feat(embsdic, probe_feat)
    pgdict = recognise_feat(probe_feat, gallery_feat)
    # pgdict1 = recognise(embsdic, probe)
    # pgdict2 = recognise(embsdic, video)
    # pgdict3 = recognise(embsdic, gallery)

    writeresult(pgdict, probe_video_path, video_save_folder)


if __name__ == "__main__":
    main()
