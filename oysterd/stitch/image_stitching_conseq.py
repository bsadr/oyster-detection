# USAGE
# python image_stitching_simple.py --images images/scottsdale --output output.png

from imutils import paths
import argparse
import cv2
import logging
from tqdm import tqdm

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

# loop over the image paths, load each one, and add them to our images to stitch list
for i, imagePath in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	images.append(image)
final = images.pop(0)
stitches = []
pbar = tqdm(total=len(images), unit=" stitches")
last_idx = 0
for i, img in enumerate(images):
	stitcher = cv2.Stitcher_create(True)
	(status, stitched) = stitcher.stitch([final, img])
	# if the status is '0', then OpenCV successfully performed image stitching
	pbar.set_description("Restart stitching at {} ".format(imagePaths[last_idx]))
	if status == 0:
		cv2.imwrite('{}/{}.jpg'.format(args["output"], i), stitched)
		final = stitched.copy()
	else:
		stitches.append(final.copy())
		final = img.copy()
		last_idx = i
	pbar.update()
pbar.close()

print("[INFO] stitching final of {}".format(len(stitches)))
stitcher = cv2.Stitcher_create(True)
(status, stitched) = stitcher.stitch(stitches)
if status == 0:
	cv2.imwrite('{}/final.jpg'.format(args["output"]), stitched)
else:
	logging.error('Failed to stitch images.')