from functions import *

M=np.eye(3)
plane=init()
while True:
    keyValue=input()
    if keyValue=='a':
        M=getTranslateMatrix(0,-5)@M
    elif keyValue=='d':
        M=getTranslateMatrix(0,5)@M
    elif keyValue=='w':
        M=getTranslateMatrix(-5,0)@M
    elif keyValue=='s':
        M=getTranslateMatrix(5,0)@M
    elif keyValue=='f':#y axis
        M=np.array([[1,0,0],[0,-1,800],[0,0,1]])@M
    elif keyValue=='g': #x axis
        M=np.array([[-1,0,800],[0,1,0],[0,0,1]])@M
    elif keyValue=='r':
        M=getRotateMatrix(5)@M
    elif keyValue=='t':
        M=getRotateMatrix(-5)@M
    elif keyValue=='x':
        M=getScaleMatrix(-5,0)@M
    elif keyValue=='c':
        M=getScaleMatrix(5,0)@M
    elif keyValue=='y':
        M=getScaleMatrix(0,-5)@M
    elif keyValue=='u':
        M=getScaleMatrix(0,5)@M
    elif keyValue=='h':
        M=np.eye(3)
        continue
    elif keyValue=='q':
        break
    else:
        continue
    Img=get_transformed_img(plane,M)
    drawAxesInImg(Img)


# plane=init()
# M=getRotateMatrix(-5*9) #9t
# M=np.dot(getTranslateMatrix(-5*30,5*20),M) #20d 30w
# plane=get_transformed_img(plane,M)
# drawAxesInImg(plane)
# M2=getScaleMatrix(-5*5,0) #5x
# M2=np.dot(getRotateMatrix(5*10),M2) #10r
# M2=np.dot(np.array([[-1,0,800],[0,1,0],[0,0,1]]),M2) #g
# plane2=get_transformed_img(plane,M2)
# drawAxesInImg(plane2)
# M3=getScaleMatrix(5*10,0) #10c
# M3=np.dot(getScaleMatrix(0,5*10),M3) #10u
# M3=np.dot(getRotateMatrix(-5*15),M3)
# plane3=get_transformed_img(plane2,M3)
# drawAxesInImg(plane3)