import tensorflow as tf
import depth.tensorflow.models.fcrn as depth
import os
from huber import huber_loss

def setup_depth_model(image_size=128, learning_rate=1e-3, reg=0.0, batch_size=1, sess=tf.Session()):
  # Create a placeholder for the input image
  model = {}
  model['X'] = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

  # Construct the network
  net = depth.ResNet50UpProj({'data': model['X']}, batch_size)
  pre_trained = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        './depth/tensorflow/NYU_ResNet-UpProj.npy')
  if not (os.path.isfile(pre_trained)):
    print("Couldn't find network...")
    exit(-1)
  net.load(pre_trained, sess)
  is_training = tf.placeholder(tf.bool)
  y = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
  y_res = tf.image.resize_images(y, [image_size / 2, image_size / 2],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  y_out = net.get_output()
  c = tf.abs(tf.reduce_max(y_out - y_res) / 5)
  loss = tf.reduce_mean(huber_loss(y_out, y_res, c))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                               beta1=0.9,
                               beta2=0.999,
                               epsilon=1e-08)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
 # Required for Batch Normalization
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
      train_step = optimizer.minimize(loss)
  uninitialized_vars = []
  for var in tf.global_variables():
      try:
          sess.run(var)
      except tf.errors.FailedPreconditionError:
          uninitialized_vars.append(var)

  init_new_vars_op = tf.variables_initializer(uninitialized_vars)
  sess.run(init_new_vars_op)

  # # Store Model in Dictionary
  model['y'] = y
  model['is_training'] = is_training
  model['y_out'] = y_out
  model['loss_val'] = loss
  model['train_step'] = train_step
  model['y_res'] = y_res

  return model
