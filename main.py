import os
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from utils.preprocess import *
from utils.generate import *
from math import ceil

## Get parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")
parser.add_argument("-b", "--block_size", type=int, default=20, help="block size in pixels")
parser.add_argument("-o", "--overlap", type=int, default=-1, help="overlap size in pixels (defaults to 1/5th of block size)")
parser.add_argument("-s", "--scale", type=float, default=4, help="Scaling w.r.t. to image size")
parser.add_argument("-n", "--num_outputs", type=int, default=1, help="number of output textures required")
parser.add_argument("-f", "--output_file", type=str, default="output{}.png", help="output file name")
parser.add_argument("-p", "--plot", type=bool, default=True, help="Show plots")
parser.add_argument("-t", "--tolerance", type=float, default=0.1, help="Tolerance fraction")

args = parser.parse_args()

if __name__ == "__main__":
	# Start the main loop here
	path = args.image_path
	block_size = args.block_size
	scale = args.scale
	overlap = args.overlap
	# Set overlap to 1/5th of block size
	if overlap < 0:
		overlap = int(block_size/6)

	# Get all blocks
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
	print("Image size: ({}, {})".format(*image.shape[:2]))

	# Find all blocks using a raster scan
	imageBlocks = rasterScan(image, block_size)
	# Find min cuts for all image block combinations
	print("Finding minCut...")
	minCutH, minCutV, errH, errV = findMinCut(imageBlocks, block_size, overlap)
	print("Found minCut...")

	H, W = image.shape[:2]
	outH, outW = int(scale*H), int(scale*W)

	for i in range(args.num_outputs):
		textureMap = generateTextureMap(imageBlocks, block_size, overlap, minCutH, minCutV, errH, errV, outH, outW, args.tolerance)
