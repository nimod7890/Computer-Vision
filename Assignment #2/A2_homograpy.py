import numpy as np
import cv2
from functions import *
import time
np.random.seed(1000)

# 2-1
img1=cv2.imread('cv_desk.png',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('cv_cover.jpg',cv2.IMREAD_GRAYSCALE)
N=30
srcP,destP,dst=getPositionMatrices(img1,img2,N)
img_show(dst)

# 2-2
H=compute_homography(srcP,destP)
transformed_img=cv2.warpPerspective(img2,H,img1.shape[::-1])
wrappedImgShow(img1,transformed_img)

# 2-3
start=time.time()
H2=compute_homography_ransac(srcP,destP,0.8)
print(f"ransac: {time.time()-start}")
transformed_img = cv2.warpPerspective(img2, H2, img1.shape[::-1])
wrappedImgShow(img1,transformed_img)

# 2-4
img3=cv2.imread('hp_cover.jpg',cv2.IMREAD_GRAYSCALE)
transformed_img=cv2.warpPerspective(cv2.resize(img3,img2.shape[::-1]),H2,img1.shape[::-1])
wrappedImgShow(img1,transformed_img)

# 2-5
img4=cv2.imread('diamondhead-10.png',cv2.IMREAD_GRAYSCALE)
img5=cv2.imread('diamondhead-11.png',cv2.IMREAD_GRAYSCALE)

srcP,destP,_=getPositionMatrices(img4,img5,N)
H3=compute_homography_ransac(srcP,destP,0.8)

h,w=img4.shape
d=int(destP[0,0]-srcP[0,0])
wrappedImg = cv2.warpPerspective(img5,H3,(w+d,h)) 
bgImg=np.column_stack((img4,np.zeros((h,d))))
stitchedImg=stitchedImgShow(bgImg,wrappedImg)
gradationImgShow(stitchedImg,wrappedImg,w)