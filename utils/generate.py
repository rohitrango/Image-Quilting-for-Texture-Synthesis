import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import ceil

ErrorCombinationFunc = np.add

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

	# Fill the first row 
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):
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
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)]*minCutH[idxI, idxJ][:, :, None] + placeBlock*(1 - minCutH[idxI, idxJ][:, :, None])


	# Fill the first column
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):
		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		idxI = blockIdxMap[i, 0]
		err = errV[idxI, :]
		minerr = err.min()
		idxJ = np.where((err <= (1.0 + tolerance)*minerr))[0]
		idxJ = np.random.choice(idxJ)
		# Place that block using minCut from minCutH
		blockIdxMap[i+1, 0] = idxJ
		placeBlock = imageBlocks[idxJ]
		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = textureMap[(blkIdx):(blkIdx+blocksize), :blocksize]*minCutV[idxI, idxJ][:, :, None] + placeBlock*(1 - minCutV[idxI, idxJ][:, :, None])

	# Fill in the other rows and columns
	for i in range(1, nH+1):
		for j in range(1, nW+1):
			# Choose the starting index for the texture placement
			blkIndexI = i*(blocksize-overlap)
			blkIndexJ = j*(blocksize-overlap)
			# Find the left and top block, and the min errors independently
			idxH = blockIdxMap[i, j-1]
			idxV = blockIdxMap[i-1, j]
			# Find errors and use a function
			err1 = errH[idxH, :]
			err2 = errV[idxV, :]
			# Find err = max(err1, err2)
			err = ErrorCombinationFunc(err1, err2)
			minerr = err.min()
			# Index of placed block
			idxPlaced = np.where((err <= (1.0 + tolerance)*minerr))[0]
			idxPlaced = np.random.choice(idxPlaced)
			# Place that block using minCut 
			blockIdxMap[i, j] = idxPlaced
			placeBlock = imageBlocks[idxPlaced]
			# Generate mask
			mask = np.maximum(minCutV[idxV, idxPlaced], minCutH[idxH, idxPlaced])
			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)]*mask[:, :, None] + placeBlock*(1 - mask[:, :, None])

	# print(textureMap.max(), textureMap.min())
	plt.imshow(textureMap)
	plt.show()
	return textureMap
