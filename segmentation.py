# -*- coding: utf-8 -*-
# segmentation.py
# this file is material part of of the dissertation 'Deep Learning for Emotion Recognition in Cartoons'
# [c] 2016-2017 John Wesley Hill

import os
import cv2
import numpy as np

# settings
videopath = './No Sound/'
exclusion = {
	'.DS_Store'
}

# gets the import location for videos (dataset).
def get_dataset(videopath=videopath):
	videos = os.listdir(os.path.dirname(videopath))
	for item in videos:
		if item in exclusion:
			videos.remove(item)
	return videos

# detect character by using a custom trained haar cascade for each character.
def detect(character, video, show_video=True):
	cap = cv2.VideoCapture(videopath + video)	
	face_cascade = cv2.CascadeClassifier(character['cascade'])
	results_path = os.path.join('results/' + "tom_or_jerry")
	
	# make a folder in results for our recognised faces.
	if not os.path.exists(results_path) and character['save'] == True:
		os.mkdir(results_path)
	
	while(1):
		# grab a frame.
		ret, frame = cap.read()
		faces = None
		
		if character['name'] == "Tom":
			# detect faces in our image.
			faces = face_cascade.detectMultiScale(frame, 
						scaleFactor=1.10, 
						minNeighbors=40, 
						minSize=(24, 24), 
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
		else:
			faces = face_cascade.detectMultiScale(frame, 
						scaleFactor=1.10, 
						minNeighbors=20, 
						minSize=(24, 24), 
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
			
		# loop over detected faces.
		for (x, y, w, h) in faces:
			# setup region of interest (ROI) for the captured face.
			roi = frame[y:y+h, x:x+w]
			
			frame_number = str(int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
			
			# write detected face to disk.
			if character['save'] == True:
				cv2.imwrite(results_path + '/' + "tom_or_jerry" + '_frame_' + frame_number + '.png', roi)
			
			if show_video is True:
				# display detection box for visual purposes.
				cv2.rectangle(frame, (x, y), (x+w, y+h), character['detect_color'], 2)
				cv2.putText(frame, character['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
			else:
				print 'detected face @ frame ' + frame_number
		
		if show_video is True:
			# display our image.
			try:
			 cv2.imshow('frame', frame)
			except:
			 break
			
			# quit or (next video) on esc.
			esc = cv2.waitKey(30) & 0xff
			if esc == 27:
				break

	# destroy & release resources.
	cv2.destroyAllWindows()
	cap.release()
	
# process all our videos.
def process(character):
	videos = get_dataset()
	print 'number of videos: ' + str(len(videos))
	
	for video in enumerate(videos):
		episode = video[1].split('- ')[1].split('(')[0].strip()
				
		# dump frames and save to disk each character.
		print 'attempting to detect ' + character['name'] +  ' in \'' + episode + '\''

		# detect our character.
		detect(character, video[1], show_video=True)

def main():
	# step 1: prepare our results folder.
	if not os.path.exists('results'):
		os.mkdir('results')

	# step 2: process all our videos to detect Tom & Jerry.
	characters = [
		{
			'name':      "Tom",
			'detect_color': (165, 91, 0),
			'save':		 True,
			'cascade':   'tom.xml'
		},
		{
			'name':      "Jerry",
			'detect_color': (165, 100, 0),
			'save':		 True,
			'cascade':   'jerry.xml'
		}
	]
	# process characters...
	[process(character) for character in characters]
	print 'done'
	
if __name__ == '__main__':
	main()