## Handles all the preprocessing
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

inf = np.float('inf')

def rasterScan(image, blocksize, step=None):
	'''
	Perform raster scan for image with squared block size "b"
	- If block size is not divisible by image size, then take all except last block
	- And for the last block, take the block from the other end
	'''
	block_list = []
	if step is None:
		step = blocksize

	H, W = image.shape[:2]
	Y = range(0, H-blocksize, step)
	X = range(0, W-blocksize, step)
	if H%step != 0:
		Y = Y[:-1]
	if W%step != 0:
		X = X[:-1]

	for y in Y:
		for x in X:
			block_list.append(image[y:y+blocksize, x:x+blocksize, :])

	print("Created {} blocks.".format(len(block_list)))
	return block_list

def VerticalOverlap(im1, im2, blocksize, overlap):
	'''
	Horizontal overlap between im1 (left) and im2 (right)
	'''
	im1Rot = np.rot90(im1)
	im2Rot = np.rot90(im2)

	mask, minVal = HorizontalOverlap(im1Rot, im2Rot, blocksize, overlap)
	mask = np.rot90(mask, 3)

	# plt.imshow(mask)
	# plt.show()
	return mask, minVal
