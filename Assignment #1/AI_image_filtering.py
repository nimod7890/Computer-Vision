from functions import *

print("========================================================")
print(f"get_gaussian_filter_1d(5,1) result: \n{get_gaussian_filter_1d(5,1)}\n")
print(f"get_gaussian_filter_2d(5,1) result: \n{get_gaussian_filter_2d(5,1)}")
print("=========================lenna==========================")
LENNA_IMG_PATH="lenna.png"
get_collage(LENNA_IMG_PATH)
print(f"lenna.png's difference: {getDifference(LENNA_IMG_PATH,17,11)}")
print("=========================shapes=========================")
SHAPE_IMG_PATH="shapes.png"
get_collage(SHAPE_IMG_PATH)
print(f"shapes.png's difference: {getDifference(SHAPE_IMG_PATH,17,11)}")
print("========================================================")
