from extractor import gaitcompare

def recognise(embsdic, probe):
    # compare
    pgdict = gaitcompare(embsdic, probe)
    print("################## probe - gallery ##################")
    print(pgdict)
    return pgdict