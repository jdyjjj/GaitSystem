import json
# 适配track_gait_writevideo3的
# def writejson(path, idlist, gallery=True):
#     with open(path,'rb')as f:
#         jsondict = json.load(f)
#     list = []
#     for frame_id in idlist:
#         tid = '{:03d}'.format(frame_id)
#         list.append(tid)
#     if gallery:
#         jsondict['TEST_SET'] = list
#     else:
#         jsondict["COMPARE_SET"] = list
#     with open(path, 'w') as write_f:
#         json.dump(jsondict, write_f, indent=4, ensure_ascii=False)

# def getid(path, id, gallery=True):
#     with open(path,'rb')as f:
#         jsondict = json.load(f)
#     return jsondict['TEST_SET'][id-1] if gallery else jsondict['COMPARE_SET'][id-1]

# def getidlist(path, gallery):
#     with open(path,'rb')as f:
#         jsondict = json.load(f)
#     return jsondict['TEST_SET'] if gallery else jsondict['COMPARE_SET']

def writejson(path, idlist, pidlist):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    list = []
    for frame_id in idlist:
        if frame_id not in pidlist: 
            tid = '{}{:03d}'.format("g", frame_id)
            list.append(tid)
    for frame_id in pidlist:
        tid = '{}{:03d}'.format("p", frame_id)
        list.append(tid)
    jsondict['TEST_SET'] = list
    with open(path, 'w') as write_f:
        json.dump(jsondict, write_f, indent=4, ensure_ascii=False)

def getid(path, id):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    return jsondict['TEST_SET'][id-1]

def getidlist(path, gallery):
    with open(path,'rb')as f:
        jsondict = json.load(f)
    return jsondict['TEST_SET'] if gallery else jsondict['COMPARE_SET']
    