import flask
import flask
from tensorflow import keras
import numpy as np
import cv2
from cv2 import cv2
import base64
import string
import sys
import logging

# app = flask.Flask(__name__)
app=flask.Flask(__name__,template_folder='templates')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

model = keras.models.load_model('./Models/aiml_model')

@app.route('/')
def home():
	return flask.render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
	draw = flask.request.form['url']

	draw = draw[22:] 
	draw_decoded = base64.b64decode(draw)
	
	image = np.asarray(bytearray(draw_decoded))
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
	resized = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
	
	vect = np.asarray(resized, dtype="uint8")
	vect = vect.reshape(1, 28, 28, 1).astype('float32')
	vect = vect/255.0

	#check if user draw anything
	if(vect.sum() == 0):
		return flask.render_template('index.html',  prediction_labels = None, prediction_percent=None)
	
	pred = model.predict(vect)[0]

	labels = ["%d" %i for i in range(0,10)] + list(string.ascii_uppercase) #list of strings 0-9 A-Z

	# Calculating top-5 highest probabilities
	pred_dict = dict(zip(labels, pred))
	pred_sorted = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:5]
	pred_labels = list(dict(pred_sorted).keys())
	pred_percent = list(dict(pred_sorted).values())
	pred_percent = [str(np.round(p*100, 3))+"%" for p in pred_percent]

	return flask.render_template('index.html', prediction_labels = pred_labels, prediction_percent = pred_percent)

if __name__ == '__main__':
	app.run(debug=True)
	