# USAGE
# python image_stitching_simple.py --images images/scottsdale --output output.png

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
print('images count: {}'.format(len(images)))
images = [[]]
stitch_images = [[]]

# loop over the image paths, load each one, and add them to our images to stitch list
size_chunk = 3
for i, imagePath in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	images[-1].append(image)
	if i % size_chunk == size_chunk-1:
		images.append([])
# initialize OpenCV's image sticher object and then perform the image stitching
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()

idx = 0

while len(images) > 0:
	for i, chunk_images in enumerate(images):
		print("[INFO] stitching images of chunk {} out of {}".format(i, len(images)))
		(status, stitched) = stitcher.stitch(chunk_images)
		# if the status is '0', then OpenCV successfully performed image stitching
		if status == 0:
			cv2.imwrite('{}/{}.jpg'.format(args["output"], idx), stitched)
			idx += 1
			stitch_images[-1].append(stitched)
			if i % size_chunk == size_chunk-1:
				stitch_images.append([])
		else:
			print("[INFO] image stitching failed ({})".format(status))
		images = stitch_images[:]
		stitch_images = [[]]
cv2.imwrite('{}/final.jpg'.format(args["output"]), images[-1])
print('images count: {} and {}'.format(len(images[-1]), len(images)))
