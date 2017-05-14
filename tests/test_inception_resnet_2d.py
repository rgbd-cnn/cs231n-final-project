import numpy as np
import tensorflow as tf
import datetime
from utilities.train import *
from models.inception_resnet import setup_resnet_inception_model
from data.cs231n.data_utils import get_CIFAR10_data

def run_inception_resnet_2d_test(device, recover, ckpt_path, prev_epochs, epochs, debug):
  # Test with CIFAR-10 Data
  data = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                          subtract_mean=True)

  # Create Model
  print("Setting up model...")
  model = setup_resnet_inception_model([32, 32, 3], 10, 1, 2, 1, learning_rate=1e-3)
  saver = tf.train.Saver()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  train_writer = tf.summary.FileWriter('inception_train_' + time, sess.graph)
  test_writer = tf.summary.FileWriter('inception_test_' + time)

  # Recover Saved Model (if available)
  if recover:
    print("Recovering model...")
    recover_model_checkpoint(sess, saver, 'checkpoints/')

  # Debug Mode: Run on Smaller Dataset
  if debug:
    data['X_train'] = data['X_train'][0:1000]
    data['y_train'] = data['y_train'][0:1000]
    data['X_val'] = data['X_val'][0:1000]
    data['y_val'] = data['y_val'][0:1000]

  # Train Model
  print("Training model...")
  train_model(device, sess, model, data['X_train'], data['y_train'], epochs=epochs,
              batch_size=64, is_training=True, log_freq=100, plot_loss=False)

  # Check Final Training Accuracy
  print("\nFinal Training Accuracy:")
  train_model(device, sess, model, data['X_train'], data['y_train'], epochs=1,
              batch_size=64, is_training=False, log_freq=100, plot_loss=False)

  # Check Validation Accuracy
  print('\nFinal Validation Accuracy:')
  train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
              batch_size=64, is_training=False, log_freq=100, plot_loss=False)

  # Save Model Checkpoint
  save_model_checkpoint(sess, saver, ckpt_path, prev_epochs + epochs)
