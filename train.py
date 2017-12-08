# -*- coding: utf-8 -*-
# train.py
# this file is material part of of the dissertation 'Deep Learning for Emotion Recognition in Cartoons'
# [c] 2016-2017 John Wesley Hill

import os
import sys
import copy
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import TensorBoard, Callback
from keras.utils import to_categorical as one_hot
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers

from vis.utils import utils
from vis.visualization import visualize_activation, visualize_saliency, get_num_filters

model_version = "v1.6"
emotions = ["happy", "angry", "surprise", "sad", "fear", "disgust"]
w, h = (60, 60)
epochs = 50

COLOR = {
	'G':'\x1B[32m',
	'R':'\x1B[31m',
	'RS':'\x1B[0m'
}

# PlotStats callback for printing custom plot stats of the model.
class PlotStats(Callback):
	def on_train_end(self, logs={}):
		# model loss plot.
		plt.plot(self.losses)
		plt.plot(self.val_losses,  color="green")
		plt.title('Learning curve for model loss')
		plt.ylabel('loss')
		plt.xlabel('epochs ({})'.format(epochs))
		plt.legend(['training', 'testing'], loc='upper left')
		plt.savefig('model_{}_loss.png'.format(model_version))
		plt.gcf().clf()
		
		# model accuracy plot.
		plt.plot(self.acc)
		plt.plot(self.val_acc,  color="green")
		plt.title('Learning curve for model accuracy'.format(epochs))
		plt.ylabel('accuracy')
		plt.xlabel('epochs ({})'.format(epochs))
		plt.legend(['training', 'testing'], loc='upper left')
		plt.savefig('model_{}_accuracy.png'.format(model_version))
		plt.gcf().clf()
		
	def on_train_begin(self, logs={}):
		self.losses = []
		self.acc = []
		self.val_acc = []
		self.val_losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))		
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))

# loads the emotion datasets and constructs them into numpy arrays 
# for training & testing for a character.
def load_emotion_data_for(character):
	DATASETS = {
		'happy': {
			'training':'datasets/' + character + '/happy/training',
			'testing':'datasets/'  + character + '/happy/testing',
		},
		'angry': {
			'training':'datasets/' + character + '/angry/training',
			'testing':'datasets/'  + character + '/angry/testing',
		},
		'surprise': {
			'training':'datasets/' + character + '/surprise/training',
			'testing':'datasets/'  + character + '/surprise/testing',
		}
	}
	emotions_training = []
	emotions_testing = []

	# training
	# append paths for happy training...
	for hd_train in os.listdir(DATASETS['happy']['training']):
		emotions_training.append(os.path.join(DATASETS['happy']['training'], hd_train))
		
	# append paths for angry training...
	for ad_train in os.listdir(DATASETS['angry']['training']):
		emotions_training.append(os.path.join(DATASETS['angry']['training'], ad_train))
	
	# Append paths for surprise training...
	for sp_train in os.listdir(DATASETS['surprise']['training']):
		emotions_training.append(os.path.join(DATASETS['surprise']['training'], sp_train))
	
	# todo: append paths for other emotions for training...
	# ...
	
	# testing
	# append paths for happy testing...
	for hd_test in os.listdir(DATASETS['happy']['testing']):
		emotions_testing.append(os.path.join(DATASETS['happy']['testing'], hd_test))
		
	# append paths for angry testing...
	for ad_test in os.listdir(DATASETS['angry']['testing']):
		emotions_testing.append(os.path.join(DATASETS['angry']['testing'], ad_test))
	
	# append paths for surprise testing...
	for sp_test in os.listdir(DATASETS['surprise']['testing']):
		emotions_testing.append(os.path.join(DATASETS['surprise']['testing'], sp_test))
		
	# todo: append paths for other emotions for testing...
	# ...
	
	data_size = len(emotions_training) // len(DATASETS.keys())
	
	# labels
	# happy labels / label 0
	happy_labels_train = np.zeros(data_size)
	happy_labels_test = np.zeros(data_size)	
	
	# angry labels / label 1 (fill with ones)
	angry_labels_train = np.zeros(data_size)
	angry_labels_train.fill(1)
	angry_labels_test = np.zeros(data_size)
	angry_labels_test.fill(1)
	
	# surprise labels / label 2 (fill with ones)
	surprise_labels_train = np.zeros(data_size)
	surprise_labels_train.fill(2)
	surprise_labels_test = np.zeros(data_size)
	surprise_labels_test.fill(2)
	
	# todo: other emotion labels / label n (fill with n's) (see the emotion array)
	# ...
	
	# append training & testing emotion labels.
	emotion_training_labels = np.append(happy_labels_train, angry_labels_train)
	emotion_training_labels = np.append(emotion_training_labels, surprise_labels_train)
	
	emotion_testing_labels = np.append(happy_labels_test, angry_labels_test)
	emotion_testing_labels = np.append(emotion_testing_labels, surprise_labels_test)
	
	print "(training) loaded {} images & {} labels for {}...".format(len(emotions_training), len(emotion_training_labels), character)
	print "(testing) loaded {} images & {} labels for {}...".format(len(emotions_testing), len(emotion_testing_labels), character)
	
	return (emotions_training, emotion_training_labels), (emotions_testing, emotion_testing_labels)

# process images into numpy for training & testing.
def process_images(fp):
	imgs = []
	for f in fp:
		img = load_img(f)
		img = img.resize((w,h), Image.ANTIALIAS)
		img = img_to_array(img) / 255
		img = img.reshape(3, w, h)
		imgs.append(img)
	return np.array(imgs)

# display an image with a or without a label in matplotlib.
def show_image(i, l=None):
	plt.imshow(array_to_img(i[0].reshape(3, w, h)))
	if l is not None:
		print "label: {}".format(emotions[np.argmax(l[0])])
	plt.axis('off')
	plt.show()

# fetches a random image from a given dataset.
# returns a numpy image, the original image and the ground truth label.
def random_image_from_dataset(i, gtl):
	ri = np.random.choice(len(i))
	numpy_img = i[ri]
	orig = array_to_img(numpy_img.reshape(3, w, h))
	numpy_img = i[ri].reshape(1, 3, w, h)
	return numpy_img, orig, gtl[ri]

# configuration before classification and training.
def setup(reproduce=True):
	# fix the seed to reproduce results in this dissertation.
	seed = 12379231
	if reproduce is True:
		np.random.seed(seed)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	
# callbacks for keras.
def load_callbacks():
	# log to tensorboard for debugging and training + testing metrics.
	if not os.path.exists('datasets/logs'):
		os.mkdir('datasets/logs')
	ps = PlotStats()
	tb = TensorBoard(log_dir='./datasets/logs', histogram_freq=1, write_graph=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
	return [tb, ps]

# main dataset loader for tom and jerry.
def load_dataset():
	tom_training, tom_testing = load_emotion_data_for("tom")
	jerry_training, jerry_testing = load_emotion_data_for("jerry")
	
	training_i = np.append(tom_training[0], jerry_training[0])
	training_l = np.append(tom_training[1], jerry_training[1])
	
	testing_i = np.append(tom_testing[0], jerry_testing[0])
	training_l = np.append(tom_testing[1], jerry_testing[1])
	
	return (training_i, training_l), (testing_i, training_l)
	
# perform training.
def load_training_and_testing_data():
	print "loading training & testing data..."
	training, testing = load_dataset()

	# process testing and training images -> numpy arrays.
	train_images = process_images(training[0])
	test_images = process_images(testing[0])
	
	# convert training and testing to one hot vectors.
	train_labels = one_hot(training[1], num_classes=6)
	test_labels = one_hot(testing[1], num_classes=6)
	
	# shuffle training data in sync for better training.
	rng = np.random.get_state()
	np.random.shuffle(train_images)
	np.random.set_state(rng)
	np.random.shuffle(train_labels)
	
	# partition dataset 80/20. (80 -> training, 20 -> testing)
	r = np.random.rand(train_images.shape[0])
	part = r < np.percentile(r, 80)
	train_images = train_images[part]
	train_labels = train_labels[part]
	test_images = test_images[-part]
	test_labels = test_labels[-part]
	
	# optionally show images and labels.
	# show_image(train_images, train_labels)
	# show_image(test_images, test_labels)
	return train_images, train_labels, test_images, test_labels

# train images and test labels.
def train(train_i, train_l, test_i, test_l, visualise, summary):
	# additional callbacks to aid training and viewing plots and visualisations.
	cb = load_callbacks()
	
	# load our cnn model.
	cnn = load_cnn_model()
	
	# begin training and save the model when finished.
	if not os.path.isfile('model_{}_.h5'.format(model_version)):
		print "training..."
		cnn.fit(train_i, train_l, epochs=epochs, batch_size=32, verbose=1, callbacks=cb, validation_data=(test_i, test_l))
		# after training, save the weights.
		cnn.save_weights('model_{}_.h5'.format(model_version))
	
	# load the weights if they exist.
	cnn.load_weights('model_{}_.h5'.format(model_version))
	
	# model evaluation. 
	loss, acc = cnn.evaluate(test_i, test_l, verbose=0)
	print "model loss {:.1f}%".format(loss)
	print "model accuracy {:.1f}%\n".format(acc)
	
	# print summary if true.
	if summary is True:
		print "summary:"
		cnn.summary()
	
	if visualise is True:
		# show at least n test results for testing.
		n = 10
		for e, i in enumerate(range(n)):
			# fetch a random image.
			i, original, gtl = random_image_from_dataset(test_i, test_l)
			plt.imshow(original)
			plt.axis('off')
			
			print "sample image: {}\n---".format(e+1)
			
			# get the predicted class and the predicted probabilities.
			pred_class, prob = (cnn.predict_classes(i, verbose=0)[0], cnn.predict(i, verbose=0).flatten())
			predicted_emotion = str(emotions[pred_class])
			ground_truth_emotion = str(emotions[np.argmax(gtl)])
			confidence_score = float(prob[pred_class] * 100)
			
			# check if the label match the prediction.
			if ground_truth_emotion is predicted_emotion:
				plt.text(3, 7, predicted_emotion.title(), fontsize=36, color="lime")
				print "image prediction: {} | confidence score: ({:.1f}%)".format(COLOR['G'] + predicted_emotion + COLOR['RS'], confidence_score)
			else:
				plt.text(3, 7, predicted_emotion.title(), fontsize=36, color="red")
				print "image prediction: {} | confidence score: ({:.1f}%)".format(COLOR['R'] + predicted_emotion + COLOR['RS'], confidence_score)	
				
			# display the closer emotion probabilities.
			for p in np.argsort(-prob):
				print "{}: {:.1f}%".format(str(emotions[p]), float(prob[p] * 100))
			
			# display the ground truth emotion.
			print "ground truth: {}\n".format(COLOR['G'] + str(ground_truth_emotion) + COLOR['RS'])
			plt.show()
			plt.gcf().clf()
			
# the main convolutional neural network architecture.
def load_cnn_model():
	# define convnet model.
	cnn = Sequential()
	
	# 3x3 convolution & 2x2 maxpooling with a input image of 60x60x3.
	cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(3, w, h), name="conv_layer_1"))
	cnn.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))

	# 3x3 convolution & 2x2 maxpooling.
	cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_layer_2'))
	cnn.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))

	# 3x3 convolution & 9x9 maxpooling.
	cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_layer_3'))
	cnn.add(MaxPooling2D(pool_size=(9, 9), name='maxpool_3'))

	# dropout 50% and flatten layer.
	cnn.add(Dropout(0.5))
	cnn.add(Flatten(name='flatten_1'))
	
	# fully connected layers and the output layer.
	cnn.add(Dense(512, activation='relu', name='fully_connected_1'))
	cnn.add(Dense(6, activation='softmax', name='output_layer'))
	o = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	cnn.compile(loss='categorical_crossentropy', optimizer=o, metrics=['accuracy'])
	
	# return the cnn model.
	return cnn

# classify an emotion from an image.
def classify_emotion_from_image(local_image):
	# classify input image, if it exists.
	if os.path.isfile('model_{}_.h5'.format(model_version)):
		print "loading model..."
		cnn = load_cnn_model()
		cnn.load_weights('model_{}_.h5'.format(model_version))
	
		# load local image.
		loaded_img = process_images(local_image)
		print "classifying..."
		
		# get the predicted class and the predicted probabilities.
		pred_class, prob = (cnn.predict_classes(loaded_img, verbose=0)[0], cnn.predict(loaded_img, verbose=0).flatten())
		predicted_emotion = str(emotions[pred_class])
		confidence_score = float(prob[pred_class] * 100)
		print "image: {}\n---".format(sys.argv[2])
		print "image prediction: {} | confidence score: ({:.1f}%)".format(COLOR['G'] + predicted_emotion + COLOR['RS'], confidence_score)
		
		# display the closer emotion probabilities.
		for p in np.argsort(-prob):
			print "{}: {:.1f}%".format(str(emotions[p]), float(prob[p] * 100))
		
		# display image.
		plt.text(3, 7, predicted_emotion.title(), fontsize=36, color="purple")
		show_image(loaded_img)
	else:
		print "unable to classify image \'{}\', model does not exist, train the network first.".format(local_image[0])

# create visualisations, requires a predefined model.
def vis(img):
	if os.path.isfile('model_{}_.h5'.format(model_version)):
		print 'loading model...'
		cnn = load_cnn_model()
		cnn.load_weights('model_{}_.h5'.format(model_version))
		
		# list all layers in loaded model.
		layer_name = "output_layer"
		layer_idx = [idx for idx, layer in enumerate(cnn.layers) if layer.name == layer_name][0]
		
		# selected layers to visualise.
		layers = ['conv_layer_1', 'conv_layer_2', 'conv_layer_3', 'output_layer']
		
		# visualise convnet visualisation for each layer, place them in a subplot.
		for layer_name in layers:
			print "Generating visualisation of {}".format(layer_name)
			layer_idx = [idx for idx, layer in enumerate(cnn.layers) if layer.name == layer_name][0]
			
			if 'conv' not in layer_name:	
				plt.figure()
				for idx, e in enumerate(emotions):
					plt.subplot(6, 6, idx + 1)
					plt.text(1, 7, '{}'.format(e))
					img = visualize_activation(cnn, layer_idx, filter_indices=idx, max_iter=750)
					img = array_to_img(img.reshape(3, w, h))
					plt.axis('off')
					plt.imshow(img)
				
				plt.suptitle('Visualisation of the Output Layer')
				plt.savefig('{}.png'.format(layer_name), bbox_inches='tight')
				plt.show()
				break
			
			filters = np.arange(get_num_filters(cnn.layers[layer_idx]))
			
			images = []
			for idx in filters:
				img = visualize_activation(cnn, layer_idx, tv_weight=0, verbose=False, filter_indices=idx, max_iter=750)
				img = array_to_img(img.reshape(3, w, h))
				images.append(img)
			
			plt.figure()
			for idx, i in enumerate(images):
				plt.subplots_adjust(wspace=0, hspace=0)
				plt.subplot(6, 6, idx + 1)
				plt.text(0, 15, 'Filter {}'.format(idx) )
				plt.axis('off')
				plt.imshow(i)
				
			plt.suptitle('Visualisation of Convolution Layer {}'.format(layer_name[len(layer_name)-1]))
			plt.savefig('{}.png'.format(layer_name), bbox_inches='tight')
			plt.show()
			
	else:
		print 'model does not exist, train the network first.'
		
def main():
	visualise_classification = False
	summary = False
	
	# -V - visualise convnet layers.
	if '-V' in sys.argv[1:]:
		vis(sys.argv[2:])
		
	# -t - train or visualise classification or print a summary of the model.
	elif '-t' in sys.argv[1:]:
		train_i, train_l, test_i, test_l = load_training_and_testing_data()
		if '-v' in sys.argv[1:]:
			visualise_classification = True
		if '-s' in sys.argv[1:]:
			summary = True
		train(train_i, train_l, test_i, test_l, visualise_classification, summary)
	
	
	# -c - classify, classifies one image from an existing model.
	elif '-c' in sys.argv[1:]:
		if os.path.isfile(sys.argv[2]):
			# load image for classification.
			loaded_img = [sys.argv[2]]
			classify_emotion_from_image(loaded_img)
		
		else:
			print 'unable to classify image \'{}\', does not exist.'.format(sys.argv[2])
	
	
	else:
		print '### Deep Learning for Emotion Recognition in Cartoons ###'
		print 'training: (and show summary or results)'
		print 'usage: train.py -t [-v|-s]\n'
		print 'classification:'
		print 'usage: train.py -c image.jpg'
		print 'visualisation:'
		print 'usage: train.py -V'
		
if __name__ == '__main__':
	# early setup
	setup(False)
	K.set_image_dim_ordering('th')
	main()