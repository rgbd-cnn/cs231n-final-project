import os
import numpy as np
import math
import matplotlib.pyplot as plt
from resnet_2d import *
from data.princeton_sunrgbd.load_data import *
from data.cs231n.data_utils import get_CIFAR10_data

# Train the Model
def train_model(device, model, X_data, labels, epochs=1,
                batch_size=64, training=False, log_freq=100, plot_loss=False):
  # Reset Network
  tf.reset_default_graph()

  # Create Placeholder Variables
  X = tf.placeholder(tf.float32, [None, 32, 32, 3])
  y = tf.placeholder(tf.int64, [None])
  is_training = tf.placeholder(tf.bool)

  # Specify Learning Rate
  learning_rate=1e-3

  # Define Output and Calculate Loss
  y_out = resnet_2d_model(X, y, is_training)
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

  # Store Model in Dictionary
  model = {}
  model['X'] = X
  model['y'] = y
  model['is_training'] = is_training
  model['y_out'] = y_out
  model['loss_val'] = mean_loss
  model['train_step'] = train_step

  with tf.device(device):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Calculate Prediction Accuracy
    correct_prediction = tf.equal(tf.argmax(model['y_out'],1), model['y'])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Shuffle Training Data
    train_indicies = np.arange(X_data.shape[0])
    np.random.shuffle(train_indicies)

    # Populate TensorFlow Variables
    variables = [model['loss_val'], correct_prediction, accuracy]
    if training:
      variables[-1] = model['train_step']

    # Iteration Counter 
    iter_cnt = 0
    losses = []
    for epoch in range(epochs):
      # Keep Track of Loss and Number of Correct Predictions
      num_correct = 0
      epoch_loss = 0

      # Iterate Over the Entire Dataset Once
      for i in range(int(math.ceil(X_data.shape[0] / batch_size))):
        # Indices for Batch
        start_idx = (i * batch_size) % X_data.shape[0]
        idx = train_indicies[start_idx:start_idx + batch_size]
        actual_batch_size = labels[i:i + batch_size].shape[0]

        # Feed Dictionary for Batch
        feed_dict = {model['X']: X_data[idx,:],
                     model['y']: labels[idx],
                     model['is_training']: training}

        # Run TF Session (Returns Loss and Correct Predictions)
        loss, corr, _ = sess.run(variables, feed_dict=feed_dict)
        num_correct += np.sum(corr)
        epoch_loss += loss

        # Print Loss and Accuracies
        if training and (iter_cnt % log_freq) == 0:
          print("Iteration {0}: Training Loss = {1:.3g} and Accuracy = {2:.2g}"\
                .format(iter_cnt + 1, loss, np.sum(corr) / float(actual_batch_size)))
        iter_cnt += 1

      # Calculate Performance
      accuracy = num_correct / float(X_data.shape[0])
      total_loss = epoch_loss / float(X_data.shape[0])
      losses.append(total_loss)

      print("Epoch {0}: Overall Loss = {1:.3g} and Accuracy = {2:.3g}"\
            .format(epoch + 1, total_loss, accuracy))

    if plot_loss:
      plt.plot(losses)
      plt.grid(True)
      plt.xlim(-10, 800)
      plt.title('Total Loss vs. Epoch')
      plt.xlabel('Epoch')
      plt.ylabel('Total Loss')
      plt.show()

  return total_loss, accuracy

# Save Checkpoint of Model
def save_model_checkpoint(session, saver, filename):
  save_path = saver.save(session, filename)
  print("Model checkpoint saved in file: %s" % save_path)

# Recover Saved Model Checkpoint
def recover_model_checkpoint(session, saver, filename):
  saver.restore(session, filename)
  print("Model restored!")

def main():
  # Suppress Annoying TensorFlow Logs
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  # Test with CIFAR-10 Data
  data = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                          subtract_mean=True)

  # Create Model
  print("Setting up model...")
  # model = setup_model()
  model = {}

  # Train Model
  print("Training model...")
  train_model('/cpu:0', model, data['X_train'], data['y_train'], epochs=1, batch_size=128,
              training=True, log_freq=100, plot_loss=False)
  print("Final Training Accuracy:")
  train_model('/cpu:0', model, data['X_train'], data['y_train'], epochs=1, batch_size=64,
              training=False, log_freq=100, plot_loss=False)
  print('Final Validation Accuracy:')
  train_model('/cpu:0', model, data['X_val'], data['y_val'], epochs=1, batch_size=64,
              training=False, log_freq=100, plot_loss=False)

main()
exit(0)
