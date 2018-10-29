# USAGE
# python detect_faces_landmarks_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
import dlib
from imutils import face_utils
# import the necessary packages
from imutils.video import VideoStream
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

predictor = dlib.shape_predictor(args["shape_predictor"])

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#print(detections.shape, range(0, detections.shape[2]))
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			print(confidence, startX, startY, endX, endY)
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			#cv2.rectangle(image, (startX, startY), (endX, endY),
			#	(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
				
			rect = dlib.rectangle(startX, startY, endX, endY)
			shape = predictor(frame, rect)
			shape = face_utils.shape_to_np(shape)
			
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# show the face number
			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			#fname = args["image"].split("\\")[-1]
			#with open("Listfile.txt", "w") as text_file:
			#	text_file.write("%s %d %d %d %d" % (fname, x, y, w, h))
				
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			
			#txt_fname = args["image"].split(".")[0]
			
			#print(txt_fname)
			#with open(txt_fname + ".txt", "w") as text_file:
			#	text_file.write("%d %d %d %d" % (startX, startY, endX, endY))

		break
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()