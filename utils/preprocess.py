## Handles all the preprocessing
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

inf = np.float('inf')

def rasterScan(image, blocksize):
	'''
	Perform raster scan for image with squared block size "b"
	- If block size is not divisible by image size, then take all except last block
	- And for the last block, take the block from the other end
	'''
	block_list = []

	H, W = image.shape[:2]
	Y = range(0, H, blocksize)
	X = range(0, W, blocksize)
	if H%blocksize != 0:
		Y[-1] = H - blocksize
	if W%blocksize != 0:
		X[-1] = W - blocksize

	for y in Y:
		for x in X:
			block_list.append(image[y:y+blocksize, x:x+blocksize, :])

	print("Created {} blocks.".format(len(block_list)))
	return block_list


def findMinCut(imageBlocks, blocksize, overlap):
	'''
	For all pairs of image blocks, find the horizontal and vertical overlaps, and store them as binary masks
	Parameters:
		imageBlocks: list of image blocks
		blocksize:   block size of image
		overlap:     overlap area

	Returns: 
		minCutH : mask for minCut horizontally
		minCutV : mask for minCut vertically
		minCutHErr : error value for horizontal
		mincutVErr : error value for vertical
	'''
	N = len(imageBlocks)
	minCutH = np.zeros((N, N, blocksize, overlap))
	minCutV = np.zeros((N, N, overlap, blocksize))

	minCutHErr = np.zeros((N, N))
	minCutVErr = np.zeros((N, N))

	rangeN = range(N)
	idProd = product(rangeN, rangeN)
	imProd = product(imageBlocks, imageBlocks)

	for (i, j), (I, J) in zip(idProd, imProd):
		hMask, eH = HorizontalOverlap(I, J, blocksize, overlap)
		vMask, eV = VerticalOverlap(I, J, blocksize, overlap)
		minCutH[i, j] = hMask
		minCutV[i, j] = vMask
		minCutHErr[i, j] = eH
		minCutVErr[i, j] = eV

	return minCutH, minCutV, minCutHErr, minCutVErr


def HorizontalOverlap(im1, im2, blocksize, overlap):
	'''
	Horizontal overlap between im1 (left) and im2 (right)
	'''
	im1Part = im1[:, -overlap:]
	im2Part = im2[:, :overlap]
	err = ((im1Part - im2Part)**2).mean(2)

	# maintain minIndex for 2nd row onwards and 
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minVal = min(E[-1])
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask = np.zeros((blocksize, overlap))
	for i in range(len(path)):
		mask[i, :path[i]] = 1

	# plt.imshow(mask)
	# plt.show()
	return mask, minVal


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
