import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.inception_resnet import inception_res_features


def two_branch_cnn(X, A, B, C, num_classes, is_training, branch1=None,
                   branch2=None, keep_prob=None, feature_op=None):

    with tf.variable_scope(branch1):
        if branch1 == "IR2d":
            tuple1 = inception_res_features(X[:, :, :, :3], A, B, C, is_training, keep_prob=keep_prob)
        elif branch1 == "IR3d":
            tuple1 = inception_res_features(X, A, B, C, is_training, keep_prob=keep_prob)
        elif branch1 == "IRd":
            tuple1 = inception_res_features(X[:, :, :, 3:], A, B, C, is_training, keep_prob=keep_prob)
        else:
            raise Exception()

    feature1, first_layer_b1 = tuple1

    with tf.variable_scope(branch2):
        if branch2 == "IR2d":
            tuple2 = inception_res_features(X[:, :, :, :3], A, B, C, is_training, keep_prob=keep_prob)
        elif branch2 == "IR3d":
            tuple2 = inception_res_features(X, A, B, C, is_training, keep_prob=keep_prob)
        elif branch2 == "IRd":
            tuple2 = inception_res_features(X[:, :, :, 3:], A, B, C, is_training, keep_prob=keep_prob)
        else:
            raise Exception()

    feature2, first_layer_b2 = tuple2

    if feature_op == "stack":
        embedding = tf.concat([feature1, feature2], 1)
        output = slim.fully_connected(embedding, num_classes, activation_fn=None)
    elif feature_op == "bn_stack":
        normalized1 = slim.batch_norm(feature1, decay=0.99, center=True, scale=True, epsilon=1e-8,
                                      activation_fn=None, is_training=is_training, trainable=True)
        normalized2 = slim.batch_norm(feature2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                                      activation_fn=None, is_training=is_training, trainable=True)
        embedding = tf.concat([normalized1, normalized2], 1)
        output = slim.fully_connected(embedding, num_classes, activation_fn=None)
    elif feature_op == "bn_add":
        normalized1 = slim.batch_norm(feature1, decay=0.99, center=True, scale=True, epsilon=1e-8,
                                      activation_fn=None, is_training=is_training, trainable=True)
        normalized2 = slim.batch_norm(feature2, decay=0.99, center=True, scale=True, epsilon=1e-8,
                                      activation_fn=None, is_training=is_training, trainable=True)
        embedding = normalized1 + normalized2
        output = slim.fully_connected(embedding, num_classes, activation_fn=None)
    elif feature_op == "naive-add":
        embedding = feature1 + feature2
        output = slim.fully_connected(embedding, num_classes, activation_fn=None)
    else:
        raise Exception()

    return output


def setup_two_branch_cnn_model(image_size, num_classes, A, B, C,
                               learning_rate=1e-3, branch1=None, branch2=None,
                               reg=0.0, keep_prob=None, feature_op=None):
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    size = [None] + image_size
    X = tf.placeholder(tf.float32, size)
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    # Define Output and Calculate Loss
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(reg)):
        y_out = two_branch_cnn(X, A, B, C, num_classes, is_training,
                               branch1=branch1, branch2=branch2,
                               keep_prob=keep_prob, feature_op=feature_op)

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(y, num_classes),
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
