import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import tensorflow as tf
#import matplotlib.plot as plt

import utils
import vgg19
import cust_vgg19
from PIL import Image
'''

img1 = utils.load_image("./test_data/tiger.jpeg")

batch1 = img1.reshape((1,224,224,3))


setup_sess = tf.Session()

images = tf.placeholder("float", [1, 224, 224, 3])
feed_dict = {images: batch1}

vgg = vgg19.Vgg19()
with tf.name_scope("content_vgg"):
	vgg.build(images)

inp_vgg = setup_sess.run({"conv1_1": vgg.conv1_1}, \
													#"conv2_1": vgg.conv2_1, \
													#"conv3_1": vgg.conv3_1, \
													#"conv4_1": vgg.conv4_1, \
													#"conv5_1": vgg.conv5_1}, \
													feed_dict = feed_dict)

setup_sess.close()

np.save('tiger_conv1_1.npy', inp_vgg["conv1_1"])
'''
tiger_conv1_1 = np.load('tiger_conv1_1.npy')
################################################

# beginning of generating new image

################################################
#img1 = utils.load_image("./test_data/tiger.jpeg")
#batch1 = img1.reshape((1,224,224,3))
#images = tf.placeholder("float", [1, 224, 224, 3])
#feed_dict = {images: batch1}
sess = tf.InteractiveSession()


cust_vgg = cust_vgg19.Vgg19()
cust_vgg.build()

cost = tf.reduce_sum(tf.square(tiger_conv1_1 - cust_vgg.conv1_1))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

tf.global_variables_initializer().run()

N = 10
costs = np.zeros(N)
sums = np.zeros(N)
for i in range(N):
	print(1.0*i/N)
	sess.run(train_step)
	costs[i] = cost.eval()
	#values = sess.run([cost, cust_vgg.inp])
	#costs[i] = values[0]
	#sums[i] = sum(sum(sum(sum(values[1]))))

inp = cust_vgg.inp.eval()

sess.close()

inp = inp.reshape(224,224,3)
inp = inp[:,:,[2,1,0]]
for i in range(0,3):
	inp[:,:,i] = inp[:,:,i] - min(inp[:,:,i].reshape((224**2,1)))
	inp[:,:,i] = 255 * np.true_divide(inp[:,:,i],max(inp[:,:,i].reshape((224**2,1)))) 

inp = Image.fromarray(inp.astype(np.uint8))
inp.save('test.jpeg')
# convert result to image
