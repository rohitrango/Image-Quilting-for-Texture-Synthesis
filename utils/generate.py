import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import ceil
from itertools import product

inf = np.float('inf')
ErrorCombinationFunc = np.add


def findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance):
	'''
	Find best horizontal match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))
	c = np.random.randint(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]


def findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance):
	'''
	Find best horizontal and vertical match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()
		rmsVal = rmsVal + ((texture[i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))
	c = np.random.randint(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]


def findPatchVertical(refBlock, texture, blocksize, overlap, tolerance):
	'''
	Find best vertical match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))
	c = np.random.randint(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]


def getMinCutPatchHorizontal(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done horizontally
	'''
	err = ((block1[:, -overlap:] - block2[:, :overlap])**2).mean(2)
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
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask = np.zeros((blocksize, blocksize, block1.shape[2]))
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1

	resBlock = np.zeros(block1.shape)
	resBlock[:, :overlap] = block1[:, -overlap:]
	resBlock = resBlock*mask + block2*(1-mask)
	# resBlock = block1*mask + block2*(1-mask)
	return resBlock


def getMinCutPatchVertical(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done vertically
	'''
	resBlock = getMinCutPatchHorizontal(np.rot90(block1), np.rot90(block2), blocksize, overlap)
	return np.rot90(resBlock, 3)


def getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap):
	'''
	Find minCut for both and calculate
	'''
	err = ((refBlockLeft[:, -overlap:] - patchBlock[:, :overlap])**2).mean(2)
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
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask1 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask1[i, :path[i]+1] = 1

	###################################################################
	## Now for vertical one
	err = ((np.rot90(refBlockTop)[:, -overlap:] - np.rot90(patchBlock)[:, :overlap])**2).mean(2)
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
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask2 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask2[i, :path[i]+1] = 1
	mask2 = np.rot90(mask2, 3)


	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

	# Put first mask
	resBlock = np.zeros(patchBlock.shape)
	resBlock[:, :overlap] = mask1[:, :overlap]*refBlockLeft[:, -overlap:]
	resBlock[:overlap, :] = resBlock[:overlap, :] + mask2[:overlap, :]*refBlockTop[-overlap:, :]
	resBlock = resBlock + (1-np.maximum(mask1, mask2))*patchBlock
	return resBlock




def generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	
	# Starting index and block
	H, W = image.shape[:2]
	randH = np.random.randint(H - blocksize)
	randW = np.random.randint(W - blocksize)

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]
	textureMap[:blocksize, :blocksize, :] = startBlock

	# Fill the first row 
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):
		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]
		patchBlock = findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = minCutPatch
	print("{} out of {} rows complete...".format(1, nH+1))


	### Fill the first column
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):
		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		refBlock = textureMap[(blkIdx-blocksize+overlap):(blkIdx+overlap), :blocksize]
		patchBlock = findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)

		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = minCutPatch

	### Fill in the other rows and columns
	for i in range(1, nH+1):
		for j in range(1, nW+1):
			# Choose the starting index for the texture placement
			blkIndexI = i*(blocksize-overlap)
			blkIndexJ = j*(blocksize-overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)]
			refBlockTop  = textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)]

			patchBlock = findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap) 

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))
		# break

	return textureMap
