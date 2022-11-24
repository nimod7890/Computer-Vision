import numpy as np
import cv2
import copy
from compute_avg_reproj_error import *

def imgShow(img,title=""):
    img=((img-img.min())/(img.max()-img.min())* 255.).astype(np.uint8) 
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def getAMatrix(srcP,destP):
    A=[]
    for p,pDot in zip(srcP,destP):
        dx,dy=pDot; x,y=p
        Ai=np.array([[x*dx,y*dx,dx,x*dy,y*dy,dy,x,y,1]])
        if len(A)==0:  A=Ai
        else:  A=np.vstack([A,Ai])
    return A

def getNormalizingMatrix(matrix):
    matrix=np.append(matrix,np.ones((len(matrix),1)),axis=1)
    mean=np.mean(matrix)
    subtractingMatrix=np.array([[1,0,-mean],[0,1,-mean],[0,0,1]])
    subtractedMatrix=subtractingMatrix@matrix.T
    maxi,mini=np.max(subtractedMatrix),np.min(subtractedMatrix)
    scalingValue=1/(maxi-mini)
    scalingMatrix1=np.array([[1,0,-mini],[0,1,-mini],[0,0,1]])
    scalingMatrix2=np.array([[scalingValue,0,0],[0,scalingValue,0],[0,0,1]]) #x,y 각각 1 => 최대 sqrt(2)

    return scalingMatrix2@scalingMatrix1@subtractingMatrix

def getNormalizedMatrix(P,matrix):
    P=np.append(P,np.ones((len(P),1)),axis=1)
    normalized=matrix@P.T
    if 0 not in normalized[-1]:
        normalized/=normalized[-1]
    return normalized[:2].T

def compute_F_raw(M):
    
    srcP,destP=M[:,:2],M[:,2:]
    A=getAMatrix(srcP,destP)
    
    U,s,V=np.linalg.svd(A)
    F=np.reshape(V[-1],(3,3))
    
    U,s,V=np.linalg.svd(F)
    s=np.diag([*s[:2],0])
    Fhat=U@s@V
    
    return Fhat/Fhat[-1,-1]

def compute_F_norm(M):
    srcP,destP=M[:,:2],M[:,2:]
    
    srcPMatrix=getNormalizingMatrix(srcP)
    normalizedSrcP=getNormalizedMatrix(srcP,srcPMatrix)
    destPMatrix=getNormalizingMatrix(destP)
    normalizedDestP=getNormalizedMatrix(destP,destPMatrix)
    
    M=np.hstack([normalizedSrcP,normalizedDestP])
    F=compute_F_raw(M)
    return destPMatrix.T@F@srcPMatrix

def compute_F_mine(M):
    th=10
    F=None
    for i in range(5000):
        randomM=M[np.random.choice(len(M),8,replace=False)]
        f=compute_F_norm(randomM)
        err=compute_avg_reproj_error(randomM,f)
        if err<th:
            F=f
            th=err
    return F

def computeEpilpolarLine(F,p):
    x,y,x1,y1=p
    return np.array([x,y,1])@F,F@np.array([x1,y1,1])

def drawEpilpolarLine(img,line,p,color):
    _,w,_=img.shape
    a,b,c=line
    p1=(0,int(-c/b))
    p2=(w,int(-(a*w+c)/b))
    img=cv2.line(img,p1,p2,color,1)
    img=cv2.circle(img,tuple(map(int,[*p])),5,color,-1)
    return img

def getResultImgs(img1,img2,M,F):
    colorList=[(255,0,0),(0,255,0),(0,0,255)]
    key=0
    while key!=113:
        base1=copy.deepcopy(img1)
        base2=copy.deepcopy(img2)
        for p,color in zip(M[np.random.choice(len(M),3,replace=False)],colorList):
            l1,m1=computeEpilpolarLine(F,p)
            result1=drawEpilpolarLine(base1,l1,p[:2],color)
            result2=drawEpilpolarLine(base2,m1,p[2:],color)
        img=cv2.hconcat([result1,result2])
        img=((img-img.min())/(img.max()-img.min())* 255.).astype(np.uint8) 
        cv2.imshow("img",img)
        key=cv2.waitKey(0)
        cv2.destroyAllWindows()

def getResult(string):
    M=np.array(np.loadtxt(f'{string}_matches.txt'))
    fList=[compute_F_raw(M),compute_F_norm(M),compute_F_mine(M)]
    errList=[compute_avg_reproj_error(M,F) for F in fList]
    minF=fList[np.argmin(errList)]
    print(f"========{string}========")
    print(f"raw: {errList[0]}")
    print(f"norm: {errList[1]}")
    print(f"mine: {errList[2]}")
    return M,minF