import torch
import numpy as np

def getemb(data):
    return data["inference_feat"]

def computedistence(x, y):
    distance = torch.sqrt(torch.sum(torch.square(x - y)))
    return distance

def compareid(data, dict, pid, threshold_value):
    probe_name = pid.split("-")[0]
    embs = getemb(data)
    min = threshold_value
    id = None
    dic={}
    for key in dict:
        if key == probe_name:
            continue
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"],value)
                    # print("########## temp ##########")
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    dic[gid] = distance
                    # print(pid, gid, distance)
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
        # dic[id] = min
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort





















# def compareid(data, dict, jsonpath, threshold_value):
#     embs = getemb(data)
#     min = threshold_value
#     id = None

#     dic={}
    
#     for key in dict:
#         disnumb = 0
#         selfmin = threshold_value
#         #加一个尝试
#         if key[0]=="p":
#             continue
#         print("############# key ###################")
#         print(key)
#         for subject in dict[key]:
#             for type in subject:
#                 for view in subject[type]:
#                     value = subject[type][view]
#                     distance = computedistence(embs["embeddings"],value)
#                     if distance.float() < min:
#                         id = key
#                         min = distance.float()
#                     if distance.float() < selfmin:
#                         selfmin = distance.float()
#                         disnumb = selfmin
#         print("############## compare id ##################")
#         print(key)
#         # 遇到p就跳过，然后把key里面的g给去掉
#         if key[0]=="p":
#             continue
#         key = key.replace("g","")
#         print(key)
#         galleryid = getid(jsonpath, int(key))
#         print(galleryid,key)
#         galleryid = galleryid.replace("g","")
#         stridlist = list(galleryid)
#         temp = ''.join(stridlist)
#         galleryidnum = int(temp)
#         dic[galleryidnum] = disnumb
#     dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
#     if id is None:
#         print("############## no id #####################")
#     print("############## distance #####################")
#     print(distance.float(), min)
#     return id, dic_sort