import tensorflow as tf
import tensorflow.contrib.slim as slim

def inception_res_C(input, counter, is_training):
  with tf.variable_scope("inc_res_C" + str(counter)):
    # Batch Normalization
    out = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)

    # ReLU Activation
    out_relu = tf.nn.relu(out)

    # Inception Path 1
    out_1 = slim.conv2d(out_relu, 192, [1,1], activation_fn=None)

    # Inception Path 2
    out_2 = slim.conv2d(out_relu, 192, [1,1], activation_fn=None)
    out_2 = slim.batch_norm(out_2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_2 = tf.nn.relu(out_2)
    out_2 = slim.conv2d(out_2, 224, [1,3], activation_fn=None)
    out_2 = slim.batch_norm(out_2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_2 = tf.nn.relu(out_2)
    out_2 = slim.conv2d(out_2, 256, [3,1], activation_fn=None)

    # Inception Path Concatenation
    out = tf.concat([out_1, out_2], axis=3)
    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out = tf.nn.relu(out)

    # BottleNeck
    out = slim.conv2d(out, 1280, [1,1], activation_fn=None)

    # Residual Addition
    output = out + out_relu

  return output

def reduction_B(input, is_training):
  with tf.variable_scope("reduction_B"):
    # Batch Normalization
    out = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)

    # ReLU Activation
    out_relu = tf.nn.relu(out)

    # Inception Path 1
    out_1 = slim.max_pool2d(out_relu, [3, 3], stride=2, padding='VALID')

    # Inception Path 2
    out_2 = slim.conv2d(out_relu, 128, [1,1], activation_fn=None)
    out_2 = slim.batch_norm(out_2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_2 = tf.nn.relu(out_2)
    out_2 = slim.conv2d(out_2, 256, [3,3], stride=2, padding='VALID', activation_fn=None)

    # Inception Path 3
    out_3 = slim.conv2d(out_relu, 128, [1,1], activation_fn=None)
    out_3 = slim.batch_norm(out_3, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_3 = tf.nn.relu(out_3)
    out_3 = slim.conv2d(out_3, 256, [3,3], stride=2, padding='VALID', activation_fn=None)

    # Inception Path 4
    out_4 = slim.conv2d(out_relu, 128, [1,1], activation_fn=None)
    out_4 = slim.batch_norm(out_4, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_4 = tf.nn.relu(out_4)
    out_4 = slim.conv2d(out_4, 256, [3,3], activation_fn=None)
    out_4 = slim.batch_norm(out_4, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_4 = tf.nn.relu(out_4)
    out_4 = slim.conv2d(out_4, 256, [3,3], stride=2, padding='VALID', activation_fn=None)

    # Inception Path Concatenation
    output = tf.concat([out_1, out_2, out_3, out_4], axis=3)

  return output

def inception_res_B(input, counter, is_training):
  with tf.variable_scope("inc_res_B" + str(counter)):
    # Batch Normalization
    out = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)

    # ReLU Activation
    out_relu = tf.nn.relu(out)

    # Inception Path 1
    out_1 = slim.conv2d(out_relu, 192, [1,1], activation_fn=None)

    # Inception Path 2
    out_2 = slim.conv2d(out_relu, 128, [1,1], activation_fn=None)
    out_2 = slim.batch_norm(out_2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_2 = tf.nn.relu(out_2)
    out_2 = slim.conv2d(out_2, 160, [1,7], activation_fn=None)
    out_2 = slim.batch_norm(out_2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_2 = tf.nn.relu(out_2)
    out_2 = slim.conv2d(out_2, 192, [7,1], activation_fn=None)

    # Inception Path Concatenation
    out = tf.concat([out_1, out_2], axis=3)

    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)
    out = tf.nn.relu(out)

    # BottleNeck
    out = slim.conv2d(out, 512, [1,1], activation_fn=None)

    # Residual Addition
    output = out + out_relu

  return output

def reduction_A(input, is_training):
  with tf.variable_scope("reduction_A"):
    # Batch Normalization
    out = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)

    # ReLU Activation
    out_relu = tf.nn.relu(out)

    # Inception Path 1
    out_1 = slim.max_pool2d(out_relu, [3, 3], stride=2, padding='VALID')

    # Inception Path 2
    out_2 = slim.conv2d(out_relu, 128, [3,3], stride=2, padding='VALID', activation_fn=None)

    # Inception Path 3
    out_3 = slim.conv2d(out_relu, 128, [1,1], activation_fn=None)
    out_3 = slim.batch_norm(out_3, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_3 = tf.nn.relu(out_3)
    out_3 = slim.conv2d(out_3, 128, [3,3], activation_fn=None)
    out_3 = slim.batch_norm(out_3, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_3 = tf.nn.relu(out_3)
    out_3 = slim.conv2d(out_3, 256, [3,3], stride=2, padding='VALID', activation_fn=None)

    # Inception Path Concatenation
    output = tf.concat([out_1, out_2, out_3], axis=3)

  return output

def inception_res_A(input, counter, is_training):
  with tf.variable_scope("inc_res_A" + str(counter)):
    # Batch Normalization
    out = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)

    # ReLU Activation
    out_relu = tf.nn.relu(out)

    # Inception Path 1
    out_1 = slim.conv2d(out_relu, 32, [1,1], activation_fn=None)

    # Inception Path 2
    out_2 = slim.conv2d(out_relu, 32, [1,1], activation_fn=None)
    out_2 = slim.conv2d(out_2, 32, [3,3], activation_fn=None)

    # Inception Path 3
    out_3 = slim.conv2d(out_relu, 32, [1,1], activation_fn=None)
    out_3 = slim.conv2d(out_3, 48, [3,3], activation_fn=None)
    out_3 = slim.batch_norm(out_3, decay=0.99, center=True, scale=True, epsilon=1e-8,
                            activation_fn=None, is_training=is_training, trainable=True)
    out_3 = tf.nn.relu(out_3)
    out_3 = slim.conv2d(out_3, 64, [3,3], activation_fn=None)

    # Inception Path Concatenation
    out = tf.concat([out_1, out_2, out_3], axis=3)

    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True)
    out = tf.nn.relu(out)

    # BottleNeck
    out = slim.conv2d(out, 128, [1,1], activation_fn=None)

    # Residual Addition
    output = out + out_relu

  return output

def stem_unit(input, is_training):
  with tf.variable_scope("stem"):
    # Convolutional Layer (3x3)
    out = slim.conv2d(input, 32, [3,3], activation_fn=None)

    # Batch Normalization
    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True, scope='bn1')

    # ReLU Activation
    first_layer = tf.nn.relu(out)

    # Convolutional Layer (3x3)
    out = slim.conv2d(first_layer, 64, [3,3], activation_fn=None)

    # Batch Normalization
    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True, scope='bn2')

    # ReLU Activation
    out = tf.nn.relu(out)

    # Max Pooling
    out = slim.max_pool2d(out, [3, 3], stride=2, padding='VALID')

    # BottleNeck
    out = slim.conv2d(out, 80, [1,1], activation_fn=None)

    # Batch Normalization
    out = slim.batch_norm(out, decay=0.99, center=True, scale=True, epsilon=1e-8,
                          activation_fn=None, is_training=is_training, trainable=True, scope='bn3')

    # ReLU Activation
    out = tf.nn.relu(out)

    # Convolutional Layer
    out = slim.conv2d(out, 128, [3,3], activation_fn=None)

  return out, first_layer


def inception_res_features(input, num_A, num_B, num_C, is_training, keep_prob=None):
    # Stem Layers
    out, first_layer = stem_unit(input, is_training)

    # Dropout
    # out = slim.dropout(out, keep_prob=keep_prob, is_training=is_training, scope='Dropout')

    # Inception-A Block
    for i in range(num_A):
        out = inception_res_A(out, i, is_training)

    # Reduction-A Block
    out = reduction_A(out, is_training)

    # Dropout
    # out = slim.dropout(out, keep_prob=keep_prob, is_training=is_training, scope='Dropout')

    # Inception-B Block
    for i in range(num_B):
        out = inception_res_B(out, i, is_training)

    # Reduction-B Block
    out = reduction_B(out, is_training)

    # Dropout
    # out = slim.dropout(out, keep_prob=keep_prob, is_training=is_training, scope='Dropout')

    # Inception-C Block
    for i in range(num_C):
        out = inception_res_C(out, i, is_training)

    # Average Pooling
    out = slim.avg_pool2d(out, [2, 2], stride=2)

    # Dropout
    out = slim.dropout(out, keep_prob=keep_prob, is_training=is_training, scope='Dropout')

    flat = slim.layers.flatten(out, scope="embedding")

    return flat, first_layer


def inception_res_model(input, num_A, num_B, num_C, num_classes, is_training, keep_prob=None):
  features = inception_res_features(input, num_A, num_B, num_C, is_training, keep_prob=keep_prob)

  # Fully Connected Layer
  output = slim.fully_connected(features, num_classes, activation_fn=None)

  return output


def setup_resnet_inception_model(image_size, num_classes, A, B, C,
                                 learning_rate=1e-3, reg=0.0, keep_prob=None):
  # Reset Network
  tf.reset_default_graph()

  # Create Placeholder Variables
  size = [None] + image_size
  X = tf.placeholder(tf.float32, size)
  y = tf.placeholder(tf.int64, [None])
  is_training = tf.placeholder(tf.bool)

  # Define Output and Calculate Loss
  with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(reg)):
    y_out = inception_res_model(X, A, B, C, num_classes, is_training, keep_prob=keep_prob)

  total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, num_classes),
                                                       logits=y_out)
  mean_loss = tf.reduce_mean(total_loss)


  if reg > 0:
    loss = mean_loss + tf.add_n(tf.losses.get_regularization_losses("reg_loss"))
  else:
    loss = mean_loss

  # Adam Optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=0.9,
                                     beta2=0.999,
                                     epsilon=1e-08)

  # Required for Batch Normalization
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
      train_step = optimizer.minimize(loss)

  # Store Model in Dictionary
  model = {}
  model['X'] = X
  model['y'] = y
  model['is_training'] = is_training
  model['y_out'] = y_out
  model['loss_val'] = loss
  model['train_step'] = train_step

  return model
