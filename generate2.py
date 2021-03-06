import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

import utils
#import vgg19
import cust_vgg19
import getopt
from PIL import Image
import tqdm
import time


def main(argv):
	style_dir = ''
	content_dir = ''
	initial_file = ''
	
	try:
		opts, args = getopt.getopt(argv, "hs:c:i:")
	except getopt.GetoptError:
		sys.exit()

	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -s <style_directory> -c <content_directory> -i <initial_image_path>'
			sys.exit()
		elif opt == '-c':
			content_dir = arg
		elif opt == '-s':
			style_dir = arg
		elif opt == '-i':
			initial_file = arg


	print 'Style directory: ', style_dir
	print 'Content directory: ', content_dir

	content_layers = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1","conv4_2"]
	style_layers = ["conv1_1_G","conv2_1_G","conv3_1_G","conv4_1_G","conv5_1_G"]

	# load content layers
	content_path = content_dir + "/"
	style_path = style_dir + "/"
	# right now, I'm only loading conv4_2
	'''
	content = {\ #content_layers[0]: np.load(content_path + content_layers[0] + ".npy"), \
						 #content_layers[1]: np.load(content_path + content_layers[1] + ".npy"), \
						 #content_layers[2]: np.load(content_path + content_layers[2] + ".npy"), \
						 #content_layers[3]: np.load(content_path + content_layers[3] + ".npy"), \
						 #content_layers[4]: np.load(content_path + content_layers[4] + ".npy"), \
						 content_layers[5]: np.load(content_path + content_layers[5] + ".npy")}
	'''
	content = { content_layers[0]: np.load(style_path + content_layers[0] + ".npy"), \
							content_layers[5]: np.load(content_path + content_layers[5] + ".npy")}

	# load style layers
	style = {style_layers[0]: np.load(style_path + style_layers[0] + ".npy"), \
					 style_layers[1]: np.load(style_path + style_layers[1] + ".npy"), \
					 style_layers[2]: np.load(style_path + style_layers[2] + ".npy"), \
					 style_layers[3]: np.load(style_path + style_layers[3] + ".npy"), \
					 style_layers[4]: np.load(style_path + style_layers[4] + ".npy")}

	
	alpha_beta_ratio = 1 * 10**-4
	beta = 1;
	alpha = alpha_beta_ratio;

	# image file location
	img1 = utils.load_image(initial_file)

	noise = np.random.normal(0,0.2,(224,224,3))
	img1_noisy = img1 + noise
	img1_tensor = tf.reshape(tf.constant(img1_noisy,dtype=tf.float32),(1,224,224,3))
    

	# Assemble network
	sess = tf.InteractiveSession()
	cust_vgg = cust_vgg19.Vgg19()
	cust_vgg.build(img1_tensor)

	#conv4_2_shape = content["conv4_2"].shape
	#conv4_2_scaling = 1.0 / (4  * conv4_2_shape[1]**4 * conv4_2_shape[3]**2)
	content_cost = alpha * 1.0 / 2.0 * tf.reduce_sum(tf.square(cust_vgg.conv4_2 - content["conv4_2"]))

	#conv1_1_G_shape = style["conv1_1_G"].shape
	conv1_1_shape = cust_vgg.conv1_1.get_shape().as_list()
	#conv1_1_G_scaling = tf.constant(1.0 / (conv1_1_G_shape[0]**2 * conv1_1_G_shape[1]**2))
	conv1_1_G_scaling = tf.constant(1.0 / (conv1_1_shape[1]**4 * conv1_1_shape[3]**2))

	#conv2_1_G_shape = style["conv2_1_G"].shape
	conv2_1_shape = cust_vgg.conv2_1.get_shape().as_list()
	conv2_1_G_scaling = tf.constant(1.0 / (conv2_1_shape[1]**4 * conv2_1_shape[3]**2))

	#conv3_1_G_shape = style["conv3_1_G"].shape
	conv3_1_shape = cust_vgg.conv3_1.get_shape().as_list()
	conv3_1_G_scaling = tf.constant(1.0 / (conv3_1_shape[1]**4 * conv3_1_shape[3]**2))

	#conv4_1_G_shape = style["conv4_1_G"].shape
	conv4_1_shape = cust_vgg.conv4_1.get_shape().as_list()
	conv4_1_G_scaling = tf.constant(1.0 / (conv4_1_shape[1]**4 * conv4_1_shape[3]**2))

	#conv5_1_G_shape = style["conv5_1_G"].shape
	conv5_1_shape = cust_vgg.conv5_1.get_shape().as_list()
	conv5_1_G_scaling = tf.constant(1.0 / (conv5_1_shape[1]**4 * conv5_1_shape[3]**2))

	style_cost_1 = tf.multiply(conv1_1_G_scaling, tf.reduce_sum(tf.square(cust_vgg.conv1_1_G - style["conv1_1_G"])))
	style_cost_2 = tf.multiply(conv2_1_G_scaling, tf.reduce_sum(tf.square(cust_vgg.conv2_1_G - style["conv2_1_G"])))
	style_cost_3 = tf.multiply(conv3_1_G_scaling, tf.reduce_sum(tf.square(cust_vgg.conv3_1_G - style["conv3_1_G"])))
	style_cost_4 = tf.multiply(conv4_1_G_scaling, tf.reduce_sum(tf.square(cust_vgg.conv4_1_G - style["conv4_1_G"])))
	style_cost_5 = tf.multiply(conv5_1_G_scaling, tf.reduce_sum(tf.square(cust_vgg.conv5_1_G - style["conv5_1_G"])))

	#style_cost = tf.add(style_cost_1, tf.add(style_cost_2,style_cost_3))
	style_layer_count = 2;
	style_cost = tf.constant(0,dtype=tf.float32)
	if style_layer_count == 1:
		style_cost =  style_cost_1
	elif style_layer_count == 2:
		style_cost =  style_cost_1 + style_cost_2
	elif style_layer_count == 3:
		style_cost =  style_cost_1 + style_cost_2 + style_cost_3
	elif style_layer_count == 4:
		style_cost =  style_cost_1 + style_cost_2 + style_cost_3 + style_cost_4
	elif style_layer_count == 5:
		style_cost =  style_cost_1 + style_cost_2 + style_cost_3 + style_cost_4 + style_cost_5
	else:
		print 'error: invalid style_layer_count'
		exit(1)

	cost = tf.add(content_cost, tf.multiply(style_cost, 1.0 / style_layer_count))
	#cost = style_cost

	tf.global_variables_initializer().run()
	
	optimizer = 'bfgs'

	if optimizer == 'bfgs':
		maxiter = 50
		train_step_bfgs = tf.contrib.opt.ScipyOptimizerInterface(cost, method='L-BFGS-B', options={'maxiter':maxiter})

		N = 500
		count = N / maxiter
		costs = np.zeros(count)

		for i in tqdm.tqdm(range(count)):
			costs[i] = cost.eval()	
			print "\n{0} iteration\tcost: {1}".format(i * maxiter,costs[i])
			'''
			print "layer 1 cost: {0}".format(style_cost_1.eval())
			print "layer 2 cost: {0}".format(style_cost_2.eval())
			print "layer 3 cost: {0}".format(style_cost_3.eval())
			print "layer 4 cost: {0}".format(style_cost_4.eval())
			print "layer 5 cost: {0}".format(style_cost_5.eval())
			'''
			save_image(cust_vgg.inp.eval(), i * maxiter)
			train_step_bfgs.minimize(sess)
		inp = cust_vgg.inp.eval()
		sess.close()

		save_image(inp, N)

	elif optimizer == 'gradient_descent':
		train_step = tf.train.GradientDescentOptimizer(0.0040*10**-3).minimize(cost)

		N = 500
		save_step = 50;
		costs = np.zeros(N)

		for i in tqdm.tqdm(range(N)):
			sess.run(train_step)
			costs[i] = cost.eval()
			#print "Percent Complete: ", 1.0 * i / N, "\nCost: ", costs[i]
			#tmp = cust_vgg.inp.eval()
			#print "mean: ", np.mean(tmp.reshape((224*224*3,1)))
			#print "std: ", np.std(tmp.reshape((224*224*3,1)))
			#print "style: ", style_cost.eval()
			#print "content: ", content_cost.eval()
			#print "Style cost: ", style_cost.eval() / style_layer_count
			#print "Content cost: ", content_cost.eval()
			if np.mod(i,save_step) == 0:
				print "{0} iteration\tcost: {1}".format(i,costs[i])
				save_image(cust_vgg.inp.eval(), i)
			

		inp = cust_vgg.inp.eval()
		sess.close()

		save_image(inp, N)


	
def save_image(inp,name):
	vgg_mean = np.array([123.68, 116.779, 103.939]); #in rgb order
	inp = inp.reshape(224,224,3)
	inp = inp[:,:,[2,1,0]]
	#inp[:,:,:] = inp[:,:,:] - min(inp[:,:,:].reshape((3*224**2,1)))
	#inp[:,:,:] = 255 * np.true_divide(inp[:,:,:],max(inp[:,:,:].reshape((3*224**2,1))))
	for i in range(0,3):
		# normalize with each layer
		#inp[:,:,i] = inp[:,:,i] - min(inp[:,:,i].reshape((224**2,1)))
		# best method
		inp[:,:,i] = inp[:,:,i] + vgg_mean[i] 
		inp[:,:,i] = 255 * np.true_divide(inp[:,:,i],max(inp[:,:,i].reshape((224**2,1))))

	inp = np.clip(inp,0,255)
	inp = Image.fromarray(inp.astype(np.uint8))
	inp.save(str(name) + '.jpeg')


	

	
if __name__ == '__main__':
	main(sys.argv[1:])
