import os
import cv2
import numpy as np 
from tqdm import tqdm

print (os.path.expanduser('~/Documents'))

REBUILD_DATA = True

class DogsVSCats():
	IMG_SIZE = 50
	CATS = ""
	DOGS = ""
	TESTING = ''
	LABELS = {}
	training_data = []
	