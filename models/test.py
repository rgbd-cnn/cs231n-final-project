import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import matplotlib.pyplot as plt
from data.cs231n.data_utils import get_CIFAR10_data

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/float(actual_batch_size)))
            iter_cnt += 1
        total_correct = correct/float(Xd.shape[0])
        total_loss = np.sum(losses)/float(Xd.shape[0])
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.xlim(-10, 800)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

def residual_layers(input, num_filters, is_training, num):
    with tf.variable_scope("res_layers" + str(num)):
        # Batch Normalization
        out = slim.batch_norm(input,
                              decay=0.999,
                              center=True,
                              scale=True,
                              epsilon=1e-8,
                              activation_fn=None,
                              is_training=is_training,
                              trainable=True)
        
        # ReLU Activation
        out = tf.nn.relu(out)
        
        # Convolutional Layer (3x3)
        out = slim.conv2d(out, num_filters, [3,3], activation_fn=None)
        out = slim.dropout(out, keep_prob=0.75, is_training=is_training)
        
        # Batch Normalization
        out = slim.batch_norm(out,
                              decay=0.999,
                              center=True,
                              scale=True,
                              epsilon=1e-8,
                              activation_fn=None,
                              is_training=is_training,
                              trainable=True)
        
        # ReLU Activation
        out = tf.nn.relu(out)
        
        # Convolutional Layer (3x3)
        out = slim.conv2d(out, num_filters, [3,3], activation_fn=None)
        out = slim.dropout(out, keep_prob=0.75, is_training=is_training)
        
        # Residual Addition
        output = out + input
        
        return output

def my_model(X,y,is_training):
    num_layers = 3
    divisions = 3
    res_layers = num_layers / divisions
    num_filters = 16

    layer = slim.conv2d(X, num_filters, [3,3], scope='conv' + str(0))
    # Create Residual Sections
    for i in range(divisions):
        # Create Residual Layers
        for j in range(res_layers):
            layer = residual_layers(layer, num_filters, is_training, res_layers * i + j)
        # Increase Filter Size
        num_filters *= 2
        
        # Pooling Convolutional Layer (Stride = 2)
        layer = slim.conv2d(layer, num_filters, [3,3], stride=[2,2], normalizer_fn=slim.batch_norm, scope='conv_pool' + str(i))
        layer = slim.dropout(layer, keep_prob=0.75, is_training=is_training)
        
    # ReLU Activation
    layer = tf.nn.relu(layer)
    
    # Average Pooling (2x2)
    layer = slim.avg_pool2d(layer, kernel_size=[2,2], stride=2)
    
    # Fully-Connected Layer
    y_out = slim.fully_connected(slim.layers.flatten(layer), 10, activation_fn=None)
    
    return y_out

def main():
    # Reset Network
    tf.reset_default_graph()

    # Create Placeholder Variables
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    learning_rate=1e-3

    # Define Output and Calculate Loss
    y_out = my_model(X,y,is_training)
    total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out)
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


    # Test with CIFAR-10 Data
    data = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                            subtract_mean=True)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess,y_out,mean_loss,X_train,y_train,1,128,100,train_step,False)

    print('Training Final')
    run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
    print('Validation Final')
    run_model(sess,y_out,mean_loss,X_val,y_val,1,64)

main()
exit(0)
