import pickle
import cv2
f2 = "undefined.pkl"
#把列表l2序列化进一个文件f2中
with open(f2,"rb") as f:
    res=pickle.load(f)
    for idx, img in enumerate(res):
        name = str(idx) + ".png"
        print(name)
        print(img)
        cv2.imwrite(name, img)

    # print(res,len(res))