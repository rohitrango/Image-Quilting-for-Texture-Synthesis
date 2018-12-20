import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import ceil

def generateTextureMap(imageBlocks, blocksize, overlap, minCutH, minCutV, errH, errV, outH, outW, tolerance):
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), imageBlocks[0].shape[2]))
	
	# Starting index and block
	startIdx = np.random.choice(range(len(imageBlocks)))
	startBlock = imageBlocks[startIdx]
	textureMap[:blocksize, :blocksize, :] = startBlock

	# Keep track of blocks placed in the image
	blockIdxMap = np.zeros((nH+1, nW+1), dtype=int)
	blockIdxMap[0, 0] = startIdx

	# Fill the first row and first column
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):
		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		idxI = blockIdxMap[0, i]
		err = errH[idxI, :]
		minerr = err.min()
		idxJ = np.where((err <= (1.0 + tolerance)*minerr))[0]
		idxJ = np.random.choice(idxJ)
		# Place that block using minCut from minCutH
		blockIdxMap[0, i+1] = idxJ
		placeBlock = imageBlocks[idxJ]
		textureMap[:blocksize, (blkIdx+overlap):(blkIdx+blocksize)] = placeBlock[:, overlap:]
		# print(textureMap[:blocksize, (blkIdx):(blkIdx+overlap)].shape)
		# print(minCutH[idxI, idxJ].shape)
		# print(placeBlock[:, :overlap].shape)
		textureMap[:blocksize, (blkIdx):(blkIdx+overlap)] = textureMap[:blocksize, (blkIdx):(blkIdx+overlap)]*minCutH[idxI, idxJ][:, :, None] + placeBlock[:, :overlap]*(1 - minCutH[idxI, idxJ][:, :, None])





	plt.imshow(textureMap)
	plt.show()
	return textureMap
