import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from huber import huber_loss

import models

def predict(model_data_path, image_path, orig):

    # Default input size
    height = 128
    width = 128
    channels = 3
    batch_size = 1
    fig2 = plt.figure(2)
    img2 = Image.open(orig)
    img2 = img2.resize([width/2, height/2], Image.NEAREST)
    img2 = np.array(img2).astype('float32')/1000
    ii = plt.imshow(img2, interpolation='nearest')
    fig2.colorbar(ii)
    plt.show(block=False)

    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

        # Evalute the network for the given image
        y_out = net.get_output()
        print(y_out)
        total_loss = tf.reduce_mean(huber_loss(y_out, img2))
        loss, pred = sess.run([total_loss, y_out], feed_dict={input_node: img})
        print("plotting results")
        print("this is the loss!!!! = ", loss)
        # Plot result
        fig = plt.figure(1)
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    parser.add_argument('orig', help='Original image depth')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths, args.orig)

    os._exit(0)

if __name__ == '__main__':
    main()





