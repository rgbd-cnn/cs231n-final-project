import os
import tensorflow as tf
import numpy as np
from models.inception_resnet import setup_resnet_inception_model
from utilities.train import *
from data.cs231n.data_utils import get_CIFAR10_data

def main():
  # Suppress Annoying TensorFlow Logs
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  # Test with CIFAR-10 Data
  data = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                          subtract_mean=True)

  # Create Model
  print("Setting up model...")
  model = setup_resnet_inception_model([32, 32, 3], 10, 1, 2, 1, learning_rate=1e-3)

  # Create Session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Define Device
  device = '/gpu:0'

  # Train Model
  print("Training model...")
  train_model(device, sess, model, data['X_train'][0:100], data['y_train'][0:100], epochs=20, batch_size=64,
              is_training=True, log_freq=100, plot_loss=False)
  print("\nFinal Training Accuracy:")
  train_model(device, sess, model, data['X_train'][0:100], data['y_train'][0:100], epochs=1, batch_size=64,
              is_training=False, log_freq=100, plot_loss=False)
  print('\nFinal Validation Accuracy:')
  train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1, batch_size=64,
              is_training=False, log_freq=100, plot_loss=False)

main()
exit(0)
