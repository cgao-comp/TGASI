import numpy as np

def LPSI_ALG(Y=np.array([]), adjM=np.array([]), weight=0.5):
    D=np.diag(adjM.sum(1))
    D_normal=np.power(D,-0.5)
    D_normal[np.isinf(D_normal)]=0
    S=np.matmul(np.matmul(D_normal, adjM), D_normal)
    d2=(1-weight)*np.matmul(np.linalg.inv((np.identity(adjM.shape[0])-weight*S)), Y)
    return d2

def features_generation(Y=np.array([]), adjM=np.array([]), weight=0.5):
    d1=Y.copy()
    d2=LPSI_ALG(Y, adjM, weight)

    V3=Y.copy()
    for i in range(len(V3)):
        if V3[i] == -1:
            V3[i] = 0
    d3=LPSI_ALG(V3, adjM, weight)

    V4=Y.copy()
    for i in range(len(V4)):
        if V4[i] == 1:
            V4[i] = 0
    d4=LPSI_ALG(V4, adjM, weight)
    
    return np.concatenate([[d1], [d2], [d3], [d4]], axis=0)
