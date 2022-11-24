import winsound
import cv2
import numpy as np
import os
import time

'''Part #1'''

def img_show(img,title=""):
    img=((img-img.min())/(img.max()-img.min())* 255.).astype(np.uint8) 
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def get_padding_size(n,k):
    p_h= (k[0]-1*(n[0]//2==0))//2
    p_w=(k[1]-1*(n[1]//2==0))//2
    return p_h,p_w

def image_padding_1d(img,kernel,row=True):
    p_h,p_w=get_padding_size(img.shape,kernel.shape)
    if row==True:
        rowFirst=np.tile(img[:,:1],(1,p_w))
        rowLast=np.tile(img[:,-1:] ,(1,p_w))
        padding=np.concatenate([rowFirst,img,rowLast],1)
    else:
        columnFirst=np.tile(img[:1,:],(p_h,1))
        columnLast=np.tile(img[-1:,:],(p_h,1))
        padding=np.concatenate([columnFirst,img,columnLast],0) 
    return padding

def image_padding_2d(img,kernel):
    padding=image_padding_1d(img,kernel,False)
    padding=image_padding_1d(padding,kernel,True)
    return padding

def cross_correlation_1d_row(img,kernel):
    paddingImg=image_padding_1d(img,kernel)
    windows=np.lib.stride_tricks.sliding_window_view(paddingImg,(kernel.shape))
    return np.einsum('ij,klij->kl',kernel,windows)

def cross_correlation_1d_column(img,kernel):
    paddingImg=image_padding_1d(img,kernel,False)
    windows=np.lib.stride_tricks.sliding_window_view(paddingImg,(kernel.shape))
    return np.einsum('ij,klij->kl',kernel,windows)

def cross_correlation_1d(img,kernel):
    if kernel.shape[0]==1:
        return cross_correlation_1d_row(img,kernel)
    else:
        return cross_correlation_1d_column(img,kernel)    

def cross_correlation_2d(img,kernel):
    img=image_padding_2d(img,kernel)
    windows=np.lib.stride_tricks.sliding_window_view(img,(kernel.shape))
    return np.einsum('ij,klij->kl',kernel,windows)

def get_gaussian_filter_1d(size,sigma):
    size//=2
    x=np.arange(-size,size+1)
    kernel=np.exp(-1*x**2/(2*sigma**2))
    kernel/=kernel.sum()
    return kernel

def get_gaussian_filter_2d(size,sigma):
    original=get_gaussian_filter_1d(size,sigma)
    return np.einsum('i,j->ji', original,original)

def get_gaussian_img(img_path,size,sigma,dimension=2):
    img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if dimension==1:
        kernel=get_gaussian_filter_1d(size,sigma)
        column_kernel=np.expand_dims(kernel,axis=1)
        row_kernel=np.expand_dims(kernel,axis=0)
        img=cross_correlation_1d(img,row_kernel)
        img=cross_correlation_1d(img,column_kernel)
        return img
    else:
        return cross_correlation_2d(img,get_gaussian_filter_2d(size,sigma))

def get_collage(img_path,sizeList=[5,11,17],sigmaList=[1,6,11],dimension=2):
    result=[]
    for i,size in enumerate(sizeList):
        tmp=[]
        for j,sigma in enumerate(sigmaList):
            img=get_gaussian_img(img_path,size,sigma,dimension)
            cv2.putText(img,f"{size}X{size} s={sigma}",(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
            img_show(img)
            if j==0:
                tmp=img
            else:
                tmp=cv2.hconcat([tmp,img])
        if i==0:
            result=tmp
        else:
            result=cv2.vconcat([result,tmp])
            
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f'./result/part_1_gaussian_filtered_{img_path}',result) 

def getDifference(img_path,size,sigma):
    start=time.time()
    img_1d=get_gaussian_img(img_path,size,sigma,1)
    print(f"1D: {time.time()-start}s")
    start=time.time()
    img_2d=get_gaussian_img(img_path,size,sigma)
    print(f"2D: {time.time()-start}s")
    difference=img_1d-img_2d
    img_show(difference)
    return np.sum(np.abs(difference))





'''Part #2'''

def compute_image_gradient(img):
    sobel_x=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float64)
    sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float64)
    d_x=cross_correlation_2d(img,sobel_x)
    d_y=cross_correlation_2d(img,sobel_y)
    magnitude=(d_x**2+d_y**2)**0.5
    np.clip(magnitude, 0, 255).astype(np.uint8)
    direction=np.arctan2(d_y,d_x)* 180. / np.pi%360
    return magnitude,direction

def get_gradient_img(img_path):
    img=get_gaussian_img(img_path,7,1.5)
    start=time.time()
    magnitude,direction=compute_image_gradient(img)
    print(f"gradient: {time.time()-start}")
    img_show(magnitude)
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f'./result/part_2_edge_raw_{img_path}',magnitude) 
    return magnitude,direction

def get_closet_angle(x,array=np.array([0,45,90,135,180,225,270,315,360])):
    return np.array([np.abs(x-angle) for angle in array]).argmin()%8

def non_maximum_suppression_dir(magnitude,direction):
    direction_mapping = lambda x: get_closet_angle(x)
    vec_direction_mapping = np.vectorize(direction_mapping) 
    mappedIdx=vec_direction_mapping(direction)
    window_idx=np.array([5,2,1,0,3,6,7,8])
    r,c=magnitude.shape
    arr=np.zeros(magnitude.shape,dtype=np.float64)
    for y in range(1,r-1):
        for x in range(1,c-1):
            tmp=magnitude[y-1:y+2,x-1:x+2].flatten()
            angle=mappedIdx[y,x]
            a_c=window_idx[angle]
            b_c=window_idx[(angle+4)%8]
            a_r=window_idx[(angle+2)%8]
            b_r=window_idx[(angle+6)%8]
            if tmp[a_r]<=tmp[4]>=tmp[b_r]: 
                if tmp[a_c]<=tmp[4]>=tmp[b_c]:
                    arr[y,x]=tmp[4]
            if tmp[a_c]<=tmp[4]>=tmp[b_c]:
                if tmp[a_r]<=tmp[4]>=tmp[b_r]: 
                    arr[y,x]=tmp[4]
            elif tmp[a_r]<tmp[4]>tmp[b_r]:   
                arr[y,x]=tmp[4]
            elif tmp[a_c]<tmp[4]>tmp[b_c]:
                arr[y,x]=tmp[4]
    return arr

def get_suppressed_mag(img_path):
    mag,direction=get_gradient_img(img_path)
    start=time.time()
    suppressed_mag=non_maximum_suppression_dir(mag,direction)
    print(f"NMS: {time.time()-start}")
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_show(suppressed_mag)
    cv2.imwrite(f'./result/part_2_edge_sup_{img_path}',suppressed_mag) 


'''Part #3'''

def compute_corner_response(img):
    sobel_x=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    I_x=cross_correlation_2d(img,sobel_x)
    I_y=cross_correlation_2d(img,sobel_y)
    window=np.ones((5,5))
    I_xx=cross_correlation_2d(I_x**2,window)
    I_yy=cross_correlation_2d(I_y**2,window)
    I_xy=cross_correlation_2d(I_x*I_y,window)
    k=0.04
    det=I_xx*I_yy-I_xy**2
    trace=I_xx+I_yy
    R=det-k*trace**2
    R=R.clip(0)
    R=(R-np.min(R))/(np.max(R)-np.min(R))
    return R

def get_corner_img(img_path):   
    img=get_gaussian_img(img_path,7,1.5)
    start=time.time()
    corner=compute_corner_response(img)
    print(f"corner response: {time.time()-start}")
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save=((corner-corner.min())/(corner.max()-corner.min())*255.).astype(np.uint8) 
    img_show(corner)
    cv2.imwrite(f'./result/part_3_corner_raw_{img_path}',save)
    return corner

def get_green_img(img_path):
    corner=get_corner_img(img_path)
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img[corner>0.1]=[0,255,0]
    img_show(img)
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f'./result/part_3_corner_bin_{img_path}',img)
    
def non_maximum_suppression_win(R):
    r,c=R.shape
    suppressed_R=np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if R[i,j]>0.1:
                rl,rr,cl,cr=max(0,i-5),min(r,i+6),max(0,j-5),min(c,j+6)
                window=R[rl:rr,cl:cr]
                maxi=np.max(window)
                if maxi==R[i,j]:
                    suppressed_R[i,j]=maxi
    return suppressed_R
    
def get_circled_img(img_path):
    img=get_gaussian_img(img_path,7,1.5)
    R=compute_corner_response(img)
    start=time.time()
    suppressed_R=non_maximum_suppression_win(R)
    print(f"nms window: {time.time()-start}")
    backgroundImg=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    backgroundImg=cv2.cvtColor(backgroundImg,cv2.COLOR_GRAY2BGR)
    for x,row in enumerate(suppressed_R):
        for y in np.where(row)[0]:
            cv2.circle(backgroundImg, (y,x), 5, (0,255,0),2)
    img_show(backgroundImg)
    directory="./result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f'./result/part_3_corner_sup_{img_path}',backgroundImg)