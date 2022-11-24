import copy,cv2,numpy as np

def init():
    img=cv2.imread('smile.png',cv2.IMREAD_GRAYSCALE)
    bg=np.full((801,801),255,dtype='uint8')
    bg[350:451,345:456]=img
    return bg

def img_show(img,title=""):
    img=((img-img.min())/(img.max()-img.min())* 255.).astype(np.uint8) 
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawAxesInImg(plane):
    img=copy.deepcopy(plane)
    img[400,:],img[:,400]=0,0
    for i in range(10):
        img[400-i,800-i],img[400+i,800-i]=0,0 #right
        img[i,400-i],img[i,400+i]=0,0 #top
    img_show(img)
    
def fillLoss(img):
    h,w=img.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            # if img[i,j]==0: continue
            summ=0
            for x,y in [[i-1,j],[i-1,j-1],[i-1,j+1],[i,j-1],
                        [i,j+1],[i+1,j],[i+1,j-1],[i+1,j+1]]:
                summ+=img[x,y]
            if summ//8<50:
                img[i,j]=0
    return img

def getTranslateMatrix(x,y):
    return np.array([[1,0,x],[0,1,y],[0,0,1]])

def getRotateMatrix(d):
    d=d*np.pi/180
    matrix1=np.array([[1,0,-400],[0,1,-400],[0,0,1]])
    matrix2=np.array([[np.cos(d),-np.sin(d),0],[np.sin(d),np.cos(d),0],[0,0,1]])
    matrix3=np.array([[1,0,400],[0,1,400],[0,0,1]])
    return np.dot(matrix3,np.dot(matrix2,matrix1))

def getScaleMatrix(y=0,x=0):
    matrix1=np.array([[1,0,-400],[0,1,-400],[0,0,1]])
    matrix2=np.array([[0.01*(100+x),0,0],[0,0.01*(100+y),0],[0,0,1]])
    matrix3=np.array([[1,0,400],[0,1,400],[0,0,1]])
    # x=np.array([[1,0,0],[0,0.95,20],[0,0,1]])
    # c=np.array([[1,0,0],[0,1.05,-20],[0,0,1]])
    # y=np.array([[0.95,0,20],[0,1,0],[0,0,1]])
    # u=np.array([[1.05,0,-20],[0,1,0],[0,0,1]])
    return np.dot(matrix3,np.dot(matrix2,matrix1))

def get_transformed_img(img,M):
    h,w=img.shape
    bg=np.full((801,801),255,dtype='uint8')
    for i in range(h):
        for j in range(w):
            if img[i,j]==255: continue
            xDot,yDot=np.einsum('ij,j',M,np.array([i,j,1]))[:2]
            if 0<=xDot<801 and 0<=yDot<801:
                bg[int(xDot),int(yDot)]=img[i,j]
    return bg



def wrappedImgShow(bg,img):
    bgImg=copy.deepcopy(bg)
    h,w=bg.shape
    for i in range(h):
        for j in range(w):
            if img[i,j]!=0:
                bgImg[i,j]=img[i,j]
    img_show(bgImg)
    
def stitchedImgShow(bg,img):
    bgImg=copy.deepcopy(bg)
    h,w=bg.shape
    for i in range(h):
        for j in range(w):
            if img[i,j]!=0 and bgImg[i,j]==0:
                bgImg[i,j]=img[i,j]
    img_show(bgImg)
    return bgImg
    
def gradationImgShow(stitched,warpped,end):
    n=100
    start=end-n
    for i in range(start,end):
        p=(i-start)/n
        stitched[:,i]=warpped[:,i]*p+stitched[:,i]*(1-p)
    img_show(stitched)
    
def hammingDistance(des1,des2):
    newDistanceList=[]
    for i,d1 in enumerate(des1):
        desList=[]
        for j, d2 in enumerate(des2):
            desList.append(np.count_nonzero((d1[:,None] & (1 << np.arange(8))) != (d2[:,None] & (1 << np.arange(8)))))
        mini=min(desList)
        newDistanceList.append([i,desList.index(mini),mini])
    return newDistanceList

def getDMatch(matches):
    DMatchList=[]
    for match in matches:
        dMatch=cv2.DMatch()
        dMatch.queryIdx,dMatch.trainIdx,dMatch.distance=map(int,match)
        DMatchList.append(dMatch)
    return DMatchList

def getPositionMatrices(img1,img2,N):
    
    orb1=cv2.ORB_create()
    orb2=cv2.ORB_create()
    
    kp1,des1=orb1.detectAndCompute(img1,None)
    kp2,des2=orb2.detectAndCompute(img2,None)
    matches=sorted(hammingDistance(des1,des2),key=lambda x:x[2])
    
    topMatches=np.array(matches[:N])
    dst=cv2.drawMatches(img1,kp1,img2,kp2,getDMatch(topMatches),None,flags=2)
    
    srcP,destP=[],[]
    for p1,p2 in zip(topMatches[:,0],topMatches[:,1]):
        destP.append([kp1[p1].pt[0],kp1[p1].pt[1]])
        srcP.append([kp2[p2].pt[0],kp2[p2].pt[1]])
        
    return np.array(srcP),np.array(destP),dst

def getNormalizingMatrix(matrix):
    ##Todo mean subtraction
    mean=np.mean(matrix)
    subtractingMatrix=np.array([[1,0,-mean],[0,1,-mean],[0,0,1]])
    ##Todo scaling
    subtractedMatrix=subtractingMatrix@matrix.T
    maxi,mini=np.max(subtractedMatrix),np.min(subtractedMatrix)
    scalingValue=1/(maxi-mini)
    scalingMatrix1=np.array([[1,0,-mini],[0,1,-mini],[0,0,1]])
    scalingMatrix2=np.array([[scalingValue,0,0],[0,scalingValue,0],[0,0,1]]) #x,y 각각 1 => 최대 sqrt(2)
    return scalingMatrix2@scalingMatrix1@subtractingMatrix

def getAMatrix(srcP,destP):
    A=[]
    for p,pDot in zip(srcP,destP):
        x,y=p; dx,dy=pDot
        Ai=np.array([[-x,-y,-1,0,0,0,x*dx,y*dx,dx],
                     [0,0,0,-x,-y,-1,x*dy,y*dy,dy]])
        if len(A)==0:  A=Ai
        else:  A=np.vstack([A,Ai])
    return np.array(A)

def compute_homography(srcP,destP):
    N=len(srcP)
    srcP1=np.append(srcP,np.ones((N,1)),axis=1)
    srcPMatrix=getNormalizingMatrix(srcP1)
    normalizedSrcP=srcPMatrix@srcP1.T
    if 0 not in normalizedSrcP[-1]:
        normalizedSrcP/=normalizedSrcP[-1]
        
    destP1=np.append(destP,np.ones((N,1)),axis=1)
    destPMatrix=getNormalizingMatrix(destP1)
    normalizedDestP=destPMatrix@destP1.T
    if 0 not in normalizedDestP[-1]:
        normalizedDestP/=normalizedDestP[-1]
    
    A=getAMatrix(normalizedSrcP[:2].T,normalizedDestP[:2].T)
    
    U,s,V=np.linalg.svd(A)
    h=np.reshape(V[-1],(3,3))
    H=np.linalg.inv(destPMatrix)@h@srcPMatrix
    return H/H[-1,-1]

def compute_homography_ransac(srcP,destP,th=0.8):
    maxi,inlierList=0,[]
    N,num=len(srcP),4
    srcP=np.append(srcP,np.ones((N,1)),axis=1) #(30,3)
    for i in range(5000):
        idxList=np.random.choice(N,num,replace=False)
        
        randomChoiceSrcP,randomChoiceDestP=srcP[idxList,:2], destP[idxList]
        H=compute_homography(randomChoiceSrcP,randomChoiceDestP)
        sampleDestP=H@srcP.T
        
        if 0 not in sampleDestP[-1]:
            sampleDestP/=sampleDestP[-1]
        else:
            continue
        sampleDestP=sampleDestP[:2].T

        tmp = []
        for j,point in enumerate(zip(destP,sampleDestP)):
            dP,sP=point
            if (np.abs(dP[0]-sP[0])<=th) and (np.abs(dP[1]-sP[1])<=th):
                tmp.append(j)

        tmpLen=len(tmp)
        if tmpLen>maxi:
            inlierList=tmp
            maxi=tmpLen
    return compute_homography(srcP[inlierList,:2],destP[inlierList])
