import tensorflow as tf
import depth.tensorflow.models.fcrn as depth
import os
from huber import huber_loss

def setup_depth_model(image_size=128, learning_rate=1e-3, reg=0.0, batch_size=1, sess=tf.Session()):
  print(image_size)
  # Create a placeholder for the input image
  input_node = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

  # Construct the network
  net = depth.ResNet50UpProj({'data': input_node}, batch_size)
  net.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './depth/tensorflow/NYU_ResNet-UpProj.npy'), sess)
  y_out = net.get_output()
  is_training = tf.placeholder(tf.bool)
  y = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
  y_res = tf.image.resize_images(y, [image_size / 2, image_size / 2],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  print(y_res)
  print(y_out)
  c = tf.reduce_max(y_out - y_res) / 5
  loss = huber_loss(y_out, y_res, c)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                               beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-08)
  train_step = optimizer.minimize(loss)

  # # Store Model in Dictionary
  model = {}
  model['X'] = input_node
  model['y'] = y
  model['is_training'] = is_training
  model['y_out'] = y_out
  model['loss_val'] = loss
  model['train_step'] = train_step

  return model
