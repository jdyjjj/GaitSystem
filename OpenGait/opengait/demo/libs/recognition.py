
from extractor import gaitcompare

def recognise(cfgs, embsdic, gids):
    # compare
    pgdict = gaitcompare(cfgs, embsdic, gids)
    print("################## probe - gallery ##################")
    print(pgdict)
    return pgdict