import tensorflow as tf
import tensorflow.contrib.slim as slim


def residual_unit(input, num_filters, counter, is_training):
    with tf.variable_scope("res_unit" + str(counter)):
        # Batch Normalization
        out = slim.batch_norm(input,
                              decay=0.99,
                              center=True,
                              scale=True,
                              epsilon=1e-8,
                              activation_fn=None,
                              is_training=is_training,
                              trainable=True)

        # ReLU Activation
        out = tf.nn.relu(out)

        # Convolutional Layer (3x3)
        out = slim.conv2d(out, num_filters, [3, 3], activation_fn=None,
                          scope='conv1')
        out = slim.dropout(out, keep_prob=0.75, is_training=is_training,
                           scope='drop1')

        # Batch Normalization
        out = slim.batch_norm(out,
                              decay=0.99,
                              center=True,
                              scale=True,
                              epsilon=1e-8,
                              activation_fn=None,
                              is_training=is_training,
                              trainable=True)

        # ReLU Activation
        out = tf.nn.relu(out)

        # Convolutional Layer (3x3)
        out = slim.conv2d(out, num_filters, [3, 3], activation_fn=None,
                          scope='conv2')
        out = slim.dropout(out, keep_prob=0.75, is_training=is_training,
                           scope='drop2')

        # Residual Addition
        output = out + input

    return output


def resnet_2d_model(X, num_classes, is_training):
    num_layers = 3
    divisions = 3
    res_layers = num_layers / divisions
    num_filters = 16

    layer = slim.conv2d(X, num_filters, [3, 3], activation_fn=None,
                        scope='conv' + str(0))
    # Create Residual Sections
    for i in range(divisions):
        # Create Residual Layers
        for j in range(res_layers):
            layer = residual_unit(layer, num_filters, res_layers * i + j,
                                  is_training)
        # Increase Filter Size
        num_filters *= 2

        # Pooling Convolutional Layer (Stride = 2)
        layer = slim.conv2d(layer, num_filters, [3, 3], stride=[2, 2],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            scope='conv_pool' + str(i))
        layer = slim.dropout(layer, keep_prob=0.75, is_training=is_training,
                             scope='drop_end')

    # ReLU Activation
    layer = tf.nn.relu(layer)

    # Average Pooling (2x2)
    layer = slim.avg_pool2d(layer, kernel_size=[2, 2], stride=2)

    # Fully-Connected Layer
    y_out = slim.fully_connected(slim.layers.flatten(layer), num_classes,
                                 activation_fn=None)

    return y_out


def setup_resnet_2d_model(image_size, num_classes, learning_rate=1e-3):
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    size = [None] + image_size
    X = tf.placeholder(tf.float32, size)
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Define Output and Calculate Loss
    y_out = resnet_2d_model(X, num_classes, is_training)
    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(y, num_classes),
        logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)

    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08)

    # Required for Batch Normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)

    # Store Model in Dictionary
    model = {}
    model['X'] = X
    model['y'] = y
    model['is_training'] = is_training
    model['y_out'] = y_out
    model['loss_val'] = mean_loss
    model['train_step'] = train_step

    return model
