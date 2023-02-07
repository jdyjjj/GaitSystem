from extractor import gaitcompare, gaitfeat_compare

def recognise(probe, embsdic):
    # compare
    pgdict = gaitcompare(probe, embsdic)
    print("################## probe - gallery ##################")
    print(pgdict)
    return pgdict

def recognise_feat(probe_feat, gallery_feat):
    # compare
    pgdict = gaitfeat_compare(probe_feat, gallery_feat)
    print("################## probe - gallery ##################")
    print(pgdict)
    return pgdict