import numpy as np
import cv2
import pyheif
from glob import glob
import os 
def load_imgs_heic(path , d_size=(1920,1080)):
	images = []
	path = path
	img_names = sorted(glob(path))
	for img_name in [img_names[0] , img_names[2]]:
		print(img_name)
		heif_file = pyheif.read(img_name)
		img = np.frombuffer(heif_file.data, dtype=np.uint8).reshape(*reversed(heif_file.size), -1)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img = cv2.resize(img , d_size)
		images.append(img)
	return images

def load_img_kitti(path="/home/henry/Desktop/torch/datasets"):
	images = []
	# Desktop/torch/datasets/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data
	path_kitti_L = os.path.join(path , "kitti/*/*/image_02/data/*.png")
	path_kitti_R = os.path.join(path , "kitti/*/*/image_03/data/*.png")
	img_name_L = sorted(glob(path_kitti_L))[:100]
	img_name_R = sorted(glob(path_kitti_R))[:100]
	for (L,R) in zip(img_name_L , img_name_R):
		img_L = cv2.imread(L)
		img_R = cv2.imread(R)
		images.append([img_L , img_R])
	return images

