import numpy as np
import cv2
import pyheif
from glob import glob

def load_imgs_heic(path , d_size=[1920,1080]):
	images = []
	path = path
	for img_name in sorted(glob(path))[:2]:
		print(img_name)
		heif_file = pyheif.read(img_name)
		img = np.frombuffer(heif_file.data, dtype=np.uint8).reshape(*reversed(heif_file.size), -1)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img = cv2.resize(img , d_size)
		images.append(img)
	return images

