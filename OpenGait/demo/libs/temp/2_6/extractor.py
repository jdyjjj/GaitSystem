import os
import os.path as osp
import pickle
import sys
import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
print(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from modeling import models
import model.baselineDemo as baselineDemo
import gait_compare as gc
from changejson import getid

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


extractor_cfgs = {  
    "gaitmodel":{ # 1
        "model_type": "BaselineDemo",
        "cfg_path": "./configs/baseline/baseline_GREW.yaml",
    },
    "path":{
        "embspath": "./opengait/demo/output/Embs/", # 3
        "whole_pkl_save_path": "./opengait/demo/output/Gaitembs/", # 2
    },
}


def loadModel(model_type, cfg_path, compare=False):
    Model = getattr(baselineDemo, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    # model.compare = compare
    return model

def gait(embsdict:dict, video):
    save_video_name = video.split("/")[-1]
    save_video_name = save_video_name.split(".")[0]
    print("========= Begin storing gait information ==========")
    gaitmodel = loadModel(**extractor_cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    cfgloader = config_loader(extractor_cfgs["gaitmodel"]["cfg_path"])
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], save_video_name)
    for inputs in loader:
        ipts = gaitmodel.inputs_pretreament(inputs)
        id = inputs[1][0] + 1 # id都是0很怪
        type = inputs[2][0] 
        view = inputs[3][0]
        savePklPath = "{}/{}/{}/{}".format(extractor_cfgs["path"]["whole_pkl_save_path"], save_video_name, type, view)
        print(savePklPath)
        if not os.path.exists(savePklPath):
            os.makedirs(savePklPath)
        savePklName = "{}/{}.pkl".format(savePklPath, inputs[3][0])
        retval, embs = gaitmodel.forward(ipts)
        pkl = open(savePklName, 'wb')
        pickle.dump(embs, pkl)
        if save_video_name not in embsdict:
            embsdict[save_video_name] = []
        ap = {}
        ap[type] = {}
        ap[type][view] = embs
        
        if len(embsdict[save_video_name]) == 0:
            embsdict[save_video_name].append(ap)
        else:
            for idx, e in enumerate(embsdict[save_video_name]):
                if str(e) == str(ap):
                    break
                elif idx == len(embsdict[save_video_name])-1:
                    embsdict[save_video_name].append(ap)            
        del ipts


def gaitcompare(embsdict:dict, probe):
    print("=========== Begin ==============")
    probe_name = probe.split("/")[-1]
    probe_name = probe_name.split(".")[0]
    print(probe_name)
    print("========= Begin comparing ==========")
    gaitmodel = loadModel(**extractor_cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    cfgloader = config_loader(extractor_cfgs["gaitmodel"]["cfg_path"])
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], probe_name)
    pg_dict = {}
    matrix_dict = {}
    for inputs in loader:
        pid = probe_name + "-" + inputs[2][0] 
        print("########### pid ############")
        print(pid)
        ipts = gaitmodel.inputs_pretreament(inputs)
        retval, embs = gaitmodel.forward(ipts)
        gid, iddict = gc.compareid(retval, embsdict, pid, 100)
        print(pid, gid)
        pg_dict[pid] = gid
        matrix_dict[pid] = iddict
    print("################## matrix_dict ##################")
    print(matrix_dict)
    return pg_dict

def extract(video):
    embsdic = {}
    if not osp.exists(extractor_cfgs["path"]["embspath"]):
        os.makedirs(extractor_cfgs["path"]["embspath"])
    embs_path = "{}{}.pkl".format(extractor_cfgs["path"]["embspath"], "embeddings")
    if osp.exists(embs_path):
        print("========= Load Embs..... ==========")
        with open(embs_path,'rb') as f:	
            embsdic = pickle.load(f)	
            print(embsdic)
            print("========= Finish Load Embs..... ==========")

    gait(embsdic, video)
    print("####### embs #########")
    print(embsdic)
    # save embslist
    pkl = open(embs_path, 'wb')
    pickle.dump(embsdic, pkl)
    






















# import os
# import os.path as osp
# import pickle
# import sys
# import shutil

# root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
# sys.path.append(root)
# from utils import config_loader
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
# print(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
# from modeling import models
# import model.baselineDemo as baselineDemo
# import gait_compare as gc
# from changejson import getid

# IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


# def loadModel(model_type, cfg_path, compare=False):
#     Model = getattr(baselineDemo, model_type)
#     # Model = getattr(models, model_type)
#     cfgs = config_loader(cfg_path)
#     model = Model(cfgs, training=False)
#     model.compare = compare
#     return model

# def gait(cfgs, embsdict:dict, gids):
#     print("========= Begin storing gait information ==========")
#     gaitmodel = loadModel(**cfgs["gaitmodel"])
#     gaitmodel.requires_grad_(False)
#     gaitmodel.eval()
#     # print("######## gaitmodel.__class__ #########")
#     # print(gaitmodel.__class__)
#     cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
#     loader = gaitmodel.get_loader(
#                 cfgloader['data_cfg'], train=False)
#     if os.path.exists(cfgs["path"]["whole_pkl_save_path"]):
#         shutil.rmtree(cfgs["path"]["whole_pkl_save_path"])
#     for inputs in loader:
#         ipts = gaitmodel.inputs_pretreament(inputs)
#         id = inputs[1][0] + 1
#         type = inputs[2][0] 
#         view = inputs[3][0]
#         print("########### gait id #################")
#         print(id)
#         print(gids)
#         if id in gids:
#             savePklPath = "{}/{}{:03d}/{}/{}".format(cfgs["path"]["whole_pkl_save_path"], "g", id, type, view)
#             id = "g" + str(id)
#         else:
#             savePklPath = "{}/{}{:03d}/{}/{}".format(cfgs["path"]["whole_pkl_save_path"], "p", id, type, view)
#             id = "p" + str(id)
#         print(savePklPath)
#         if not os.path.exists(savePklPath):
#             os.makedirs(savePklPath)
#         savePklName = "{}/{}.pkl".format(savePklPath, inputs[3][0])
#         retval, embs = gaitmodel.forward(ipts)
#         pkl = open(savePklName, 'wb')
#         pickle.dump(embs, pkl)
#         if id not in embsdict:
#             embsdict[id] = []
#         ap = {}
#         ap[type] = {}
#         ap[type][view] = embs
        
#         if len(embsdict[id]) == 0:
#             embsdict[id].append(ap)
#         else:
#             for idx, e in enumerate(embsdict[id]):
#                 if str(e) == str(ap):
#                     break
#                 elif idx == len(embsdict[id])-1:
#                     embsdict[id].append(ap)            
#         del ipts

# def gaitcompare(cfgs, embsdict:dict, gids):
#     print("========= Begin comparing..... ==========")
#     gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
#     gaitmodel.requires_grad_(False)
#     gaitmodel.eval()
#     cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
#     loader = gaitmodel.get_loader(
#                 cfgloader['data_cfg'], train=False)
#     pg_dict = {}
#     matrix_dict = {}
#     for inputs in loader:
#         pid = inputs[1][0]+1
#         if pid in gids:
#             continue
#         realpid = getid(cfgs["path"]["jsonpath"], pid)
#         ipts = gaitmodel.inputs_pretreament(inputs)
#         retval, embs = gaitmodel.forward(ipts)
#         gid, iddict = gc.compareid(retval,embsdict,cfgs["path"]["jsonpath"], 100)
#         gid = gid.replace("g","")
#         print(gid)
#         realgid = getid(cfgs["path"]["jsonpath"], int(gid))
#         realgid = realgid.replace("g","")
#         stridlist = list(realgid)
#         temp = ''.join(stridlist)
#         galleryidnum = int(temp)
#         realpid = realpid.replace("p","")
#         stridlist = list(realpid)
#         temp = ''.join(stridlist)
#         probeidnum = int(temp)
#         pg_dict[probeidnum] = galleryidnum
#         matrix_dict[probeidnum] = iddict
#     print("################## matrix_dict ##################")
#     print(matrix_dict)
#     return pg_dict

# def extract(cfgs, gids):
#     embsdic = {}
#     if not osp.exists(cfgs["path"]["embspath"]):
#         os.makedirs(cfgs["path"]["embspath"])
#     embs_path = "{}{}.pkl".format(cfgs["path"]["embspath"], "embeddings")
#     gait(cfgs, embsdic, gids)
#     print("####### embs #########")
#     print(embsdic)
#     # save embslist
#     pkl = open(embs_path, 'wb')
#     pickle.dump(embsdic, pkl)
#     return embsdic





# def loadModel(model_type, cfg_path, compare=False):
#     Model = getattr(baselineDemo, model_type)
#     cfgs = config_loader(cfg_path)
#     model = Model(cfgs, training=False)
#     model.compare = compare
#     return model

# def gait(cfgs, embsdict:dict, video):
#     save_video_name = video.split("/")[-1]
#     save_video_name = save_video_name.split(".")[0]
#     print("========= Begin storing gait information ==========")
#     gaitmodel = loadModel(**cfgs["gaitmodel"])
#     gaitmodel.requires_grad_(False)
#     gaitmodel.eval()
#     cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
#     loader = gaitmodel.get_loader(
#                 cfgloader['data_cfg'], save_video_name)
#     for inputs in loader:
#         ipts = gaitmodel.inputs_pretreament(inputs)
#         id = inputs[1][0] + 1 # id都是0很怪
#         type = inputs[2][0] 
#         view = inputs[3][0]
#         savePklPath = "{}/{}/{}/{}".format(cfgs["path"]["whole_pkl_save_path"], save_video_name, type, view)
#         print(savePklPath)
#         if not os.path.exists(savePklPath):
#             os.makedirs(savePklPath)
#         savePklName = "{}/{}.pkl".format(savePklPath, inputs[3][0])
#         retval, embs = gaitmodel.forward(ipts)
#         pkl = open(savePklName, 'wb')
#         pickle.dump(embs, pkl)
#         if save_video_name not in embsdict:
#             embsdict[save_video_name] = []
#         ap = {}
#         ap[type] = {}
#         ap[type][view] = embs
        
#         if len(embsdict[save_video_name]) == 0:
#             embsdict[save_video_name].append(ap)
#         else:
#             for idx, e in enumerate(embsdict[save_video_name]):
#                 if str(e) == str(ap):
#                     break
#                 elif idx == len(embsdict[save_video_name])-1:
#                     embsdict[save_video_name].append(ap)            
#         del ipts


# def gaitcompare(cfgs, embsdict:dict, probe):
#     print("=========== Begin ==============")
#     probe_name = probe.split("/")[-1]
#     probe_name = probe_name.split(".")[0]
#     print(probe_name)
#     print("========= Begin comparing ==========")
#     gaitmodel = loadModel(**cfgs["gaitmodel"])
#     gaitmodel.requires_grad_(False)
#     gaitmodel.eval()
#     cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
#     loader = gaitmodel.get_loader(
#                 cfgloader['data_cfg'], probe_name)
#     pg_dict = {}
#     matrix_dict = {}
#     for inputs in loader:
#         pid = probe_name + "-" + inputs[2][0] 
#         print("########### pid ############")
#         print(pid)
#         ipts = gaitmodel.inputs_pretreament(inputs)
#         retval, embs = gaitmodel.forward(ipts)
#         gid, iddict = gc.compareid(retval, embsdict, pid, 100)
#         print(pid, gid)
#         pg_dict[pid] = gid
#         matrix_dict[pid] = iddict
#     print("################## matrix_dict ##################")
#     print(matrix_dict)
#     return pg_dict

# def extract(cfgs, video):
#     embsdic = {}
#     if not osp.exists(cfgs["path"]["embspath"]):
#         os.makedirs(cfgs["path"]["embspath"])
#     embs_path = "{}{}.pkl".format(cfgs["path"]["embspath"], "embeddings")
#     if osp.exists(embs_path):
#         print("========= Load Embs..... ==========")
#         with open(embs_path,'rb') as f:	
#             embsdic = pickle.load(f)	
#             print(embsdic)
#             print("========= Finish Load Embs..... ==========")

#     gait(cfgs, embsdic, video)
#     print("####### embs #########")
#     print(embsdic)
#     # save embslist
#     pkl = open(embs_path, 'wb')
#     pickle.dump(embsdic, pkl)
    