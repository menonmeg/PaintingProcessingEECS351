import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import sys
import numpy as np
import tensorflow as tf
import getopt
#import matplotlib.plot as plt

import utils
import vgg19
import cust_vgg19

def main(argv):
	input_file = ''
	output_dir = ''
	try:
		opts, args = getopt.getopt(argv,'hi:d:')
	except getopt.GetoptError:
		print 'save_content_layer.py -i <inputfile> -d <outputdirectory>'
		sys.exit(2)
	for op, arg in opts:
		if op == '-h':
			print 'save_content_layer.py -i <inputfile> -d <outputdirectory>'
			sys.exit()
		elif op == '-i':
			input_file = arg
		elif op == '-d':
			output_dir = arg

	print 'Input file: ', input_file
	print 'Output directory: ', output_dir

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	img = utils.load_image(input_file)
	batch1 = img.reshape((1,224,224,3))

	setup_sess = tf.Session()
	images = tf.placeholder("float", [1,224,224,3])
	feed_dict = {images:batch1}

	vgg = vgg19.Vgg19()
	with tf.name_scope("content_vgg"):
		vgg.build(images)

	layer_count = 5;
	layer_names = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1","conv1_1_G", "conv2_1_G", "conv3_1_G", "conv4_1_G","conv5_1_G","conv4_2"]

	layers = setup_sess.run({layer_names[0]: vgg.conv1_1, \
				layer_names[1]: vgg.conv2_1, \
				layer_names[2]: vgg.conv3_1, \
				layer_names[3]: vgg.conv4_1, \
				layer_names[4]: vgg.conv5_1, \
				layer_names[5]: vgg.conv1_1_G, \
				layer_names[6]: vgg.conv2_1_G, \
				layer_names[7]: vgg.conv3_1_G, \
				layer_names[8]: vgg.conv4_1_G, \
				layer_names[9]: vgg.conv5_1_G, \
				layer_names[10]: vgg.conv4_2}, \
				feed_dict = feed_dict)
	setup_sess.close()

	for i in range(0,layer_count):
		print i
		f_name = output_dir + "/" + layer_names[i] + ".npy"
		np.save(f_name, layers[layer_names[i]])

	


if __name__ == "__main__":
	main(sys.argv[1:])
