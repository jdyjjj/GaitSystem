
from extractor import gaitcompare

def recognise(cfgs, embsdic, probe):
    # compare
    pgdict = gaitcompare(cfgs, embsdic, probe)
    print("################## probe - gallery ##################")
    print(pgdict)
    return pgdict