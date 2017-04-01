import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

import utils
#import vgg19
import cust_vgg19
from PIL import Image

def main(argv):
	style_dir = ''
	content_dir = ''
	
	try:
		opts, args = getopt.getopt(argv, "hs:c:")
	except getopt.GetoptError:
		print 'test.py -s <style_directory> -c <content_directory>'
		sys.exit()

	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -s <style_directory> -c <content_directory>'
			sys.exit()
		elif opt == '-c':
			content_dir = arg
		elif opt == '-s':
			style_dir = arg

	print 'Style directory: ', style_dir
	print 'Content directory: ', content_dir

	content_layers = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1","conv4_2"]
	style_layers = ["conv1_1_G","conv2_1_G","conv3_1_G","conv4_1_G","conv5_1_G"]

	# load content layers
	content_path = content_dir + "/"
	# right now, I'm only loading conv4_2
	content = {\ #content_layers[0]: np.load(content_path + content_layers[0] + ".npy"), \
						 #content_layers[1]: np.load(content_path + content_layers[1] + ".npy"), \
						 #content_layers[2]: np.load(content_path + content_layers[2] + ".npy"), \
						 #content_layers[3]: np.load(content_path + content_layers[3] + ".npy"), \
						 #content_layers[4]: np.load(content_path + content_layers[4] + ".npy"), \
						 content_layers[5]: np.load(content_path + content_layers[5] + ".npy")}

	sys.exit()
	# load style layers
	style_path = style_dir + "/"
	style = {style_layers[0]: np.load(style_path + style_layers[0] + ".npy"), \
					 style_layers[1]: np.load(style_path + style_layers[1] + ".npy"), \
					 style_layers[2]: np.load(style_path + style_layers[2] + ".npy"), \
					 style_layers[3]: np.load(style_path + style_layers[3] + ".npy"), \
					 style_layers[4]: np.load(style_path + style_layers[4] + ".npy")}

	
	alpha_beta_ratio = 1 * 10**-3;
	beta = 1;
	alpha = alpha_beta_ratio;


	# Assemble network
	cust_vgg = cust_vgg19.Vgg19()
	cust_vgg.build()

	cost = alpha * tf.reduce_sum(tf.square(cust_vgg.conv4_2 - content["conv4_2"]))

	layers_to_include = 3

	cost = beta * (tf.reduce_sum(tf.square(cust_vgg.conv1_1_g - style["conv1_1_G"])) \
								+ tf.reduce_sum(tf.square(cust_vgg.conv2_1_G - style["conv2_1_G"])) |
								+ tf.reduce_sum(tf.square(cust_vgg.conv3_1_G - style["conv3_1_G"])))

	train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

	tf.global_variables_initializer().run()

	N = 10
	costs = np.zeros(N)

	for i in range(N):
		sess.run(train_step)
		costs[i] = cost.eval()
		print "Percent Complete: ", 1 * i / N, "\nCost: ", costs[i]
		

	inp = cust_vgg.inp.eval()
	sess.close()

	inp = inp.reshape(224,224,3)
	inp = inp[:,:,[2,1,0]]
	for i in range(0,3):
		inp[:,:,i] = inp[:,:,i] - min(inp[:,:,i].reshape((224**2,1)))
		inp[:,:,i] = 255 * np.true_divide(inp[:,:,i],max(inp[:,:,i].reshape((224**2,1)))) 

	inp = Image.fromarray(inp.astype(np.uint8))
	inp.save('test.jpeg')


	


	

	
if __name__ == '__main__':
	main(sys.argv[1:])
