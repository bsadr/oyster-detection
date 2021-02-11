import cv2
import os
from tqdm import tqdm

def extractVideo(video_path, extract_path, fps=3):
	video = cv2.VideoCapture(video_path)
	os.makedirs(extract_path, exist_ok=True)
	fps = min(int(video.get(cv2.CAP_PROP_FPS)), fps)
	frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=int(frame_count/fps), unit=" frames")
	frame_number = 1
	while video.isOpened() and frame_number < frame_count:
		pbar.set_description(f"Iterating {video_path}, frame {frame_number} of {frame_count}")
		success, frame = video.read()
		if success and frame_number % fps == 0:
			cv2.imwrite(f"{os.path.join(extract_path, str(frame_number).zfill(3))}.jpg", frame)
		frame_number += 1
	print(f"{video_path} -- extracted to -- {extract_path}")


if __name__ == '__main__':
	video_path = ['/scratch2/bsadrfa/camera_calibration/data/GH011030.MP4',]
#								'/scratch2/bsadrfa/camera_calibration/data/GH011023.MP4',
#								'/scratch2/bsadrfa/camera_calibration/data/GH011024.MP4',
#								'/scratch2/bsadrfa/camera_calibration/data/GH011027.MP4',
#								'/scratch2/bsadrfa/camera_calibration/data/GH011028.MP4',
#								'/scratch2/bsadrfa/camera_calibration/data/GH011022.MP4',]

	extract_path = ['/scratch2/bsadrfa/camera_calibration/data/GH011030/',]
#									'/scratch2/bsadrfa/camera_calibration/data/GH011023/',
#									'/scratch2/bsadrfa/camera_calibration/data/GH011024/',
#									'/scratch2/bsadrfa/camera_calibration/data/GH011027/',
#									'/scratch2/bsadrfa/camera_calibration/data/GH011028/',
#									'/scratch2/bsadrfa/camera_calibration/data/GH011022/',]
	for (v, e) in zip(video_path, extract_path):
		extractVideo(v, e)	
