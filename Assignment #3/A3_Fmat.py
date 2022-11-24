from functions import *
import cv2
import random

random.seed(100)
M,F=getResult("temple")
temple1=cv2.imread('temple1.png')
temple2=cv2.imread('temple2.png')
getResultImgs(temple1,temple2,M,F)

M,F=getResult("library")
library1=cv2.imread('library1.jpg')
library2=cv2.imread('library2.jpg')
getResultImgs(library1,library2,M,F)

M,F=getResult("house")
house1=cv2.imread('house1.jpg')
house2=cv2.imread('house2.jpg')
getResultImgs(house1,house2,M,F)