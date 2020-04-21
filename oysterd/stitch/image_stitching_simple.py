# USAGE
# python image_stitching_simple.py --images images/scottsdale --output output.png

from imutils import paths
import argparse
import cv2
import logging

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
print('[INFO] total images: {}'.format(len(imagePaths)))
images = []
stitch_images = []

# loop over the image paths, load each one, and add them to our images to stitch list
size_chunk = 5
for i, imagePath in enumerate(imagePaths):
	if i % size_chunk == 0:
		images.append([])
	image = cv2.imread(imagePath)
	images[-1].append(image)
idx = 0
while len(images) > 0:
	for i, chunk_images in enumerate(images):
		stitcher = cv2.Stitcher_create(True)
		print("[INFO] stitching images of chunk {} out of {}".format(i, len(images)))
		(status, stitched) = stitcher.stitch(chunk_images)
		# if the status is '0', then OpenCV successfully performed image stitching
		if status == 0:
			if i % size_chunk == 0:
				stitch_images.append([])
			cv2.imwrite('{}/{}.jpg'.format(args["output"], idx), stitched)
			idx += 1
			stitch_images[-1].append(stitched.copy())
		else:
			logging.warning("image stitching failed ({})".format(status))
	g = input("Enter>")
	images = stitch_images[:]
	stitch_images = []
if len(images) == 1:
	cv2.imwrite('{}/final.jpg'.format(args["output"]), images[-1])
else:
	logging.error('Failed to stitch images.')
print('[INFO] total of {} stitches'.format(idx))
