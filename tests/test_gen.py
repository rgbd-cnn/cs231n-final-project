import os
import shutil
import tensorflow as tf
from models.depth_prediction import setup_depth_model
from utilities.train import *


def run_gen_test(data, num_classes, device, recover, ckpt_path, prev_epochs, epochs, lr=1e-3,
                              train_epochs_per_validation=1, tensorboard_log_dir=None, dataset=None, reg=0.0):
  # Create Model
  print("Setting up model...")
  H, W, _ = data['X_train'][0].shape

  # divide by 1000 to become meters
  data['y_train'] = data['X_train'][:, :, :, 3:4] / 1000
  data['y_val'] = data['X_val'][:, :, :, 3:4] / 1000
  data['y_test'] = data['X_test'][:, :, :, 3:4] / 1000

  data['X_train'] = data['X_train'][:, :, :, 0:3]
  data['X_val'] = data['X_val'][:, :, :, 0:3]
  data['X_test'] = data['X_test'][:, :, :, 0:3]

  if (H != W):
    print("This will not work, exiting...")
    exit(-1)
  sess = tf.Session()
  # setup_depth_model(image_size=128, learning_rate=1e-3, reg=0.0, batch_size=1, sess=tf.Session())
  model = setup_depth_model(image_size=H, learning_rate=lr, batch_size=1, sess=sess)
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())

  # Recover Saved Model (if available)
  if recover:
    print("Recovering model...")
    recover_model_checkpoint(sess, saver, 'checkpoints/')

  num_train_val_cycles = epochs / train_epochs_per_validation

  if tensorboard_log_dir:
    train_dir = os.path.join(os.path.expanduser(tensorboard_log_dir), "IR-%s-lr-%s-reg-%s-train" %
      (dataset, lr, reg))
    val_dir = os.path.join(os.path.expanduser(tensorboard_log_dir), "IR-%s-lr-%s-reg-%s-val" %
      (dataset, lr, reg))

    if os.path.exists(train_dir):
      shutil.rmtree(train_dir)

    if os.path.exists(val_dir):
      shutil.rmtree(val_dir)

    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(val_dir)

  else:
    train_writer = None
    val_writer = None

  global_step = 0

  print("Training model...")
  for i in range(num_train_val_cycles):
    # Train Model
    train_gen_model(device, sess, model, data['X_train'], data['y_train'],
                epochs=train_epochs_per_validation, batch_size=1, is_training=True, log_freq=100,
                plot_loss=False, global_step=global_step,
                writer=train_writer)

    global_step += train_epochs_per_validation - 1

    # Validate Model
    if tensorboard_log_dir:
      print("Validating model...")
      train_gen_model(device, sess, model, data['X_val'], data['y_val'], epochs=1, batch_size=64,
                      is_training=False, log_freq=100, plot_loss=False, global_step=global_step,
                      writer=val_writer)

    global_step += 1

  # Check Final Training Accuracy
  print("\nFinal Training Accuracy:")
  train_gen_model(device, sess, model, data['X_train'], data['y_train'], epochs=1,
              batch_size=1, is_training=False, log_freq=100, plot_loss=False)

  # Check Validation Accuracy
  print('\nFinal Validation Accuracy:')
  train_gen_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
              batch_size=1, is_training=False, log_freq=100, plot_loss=False)

  # Check Test Accuracy
  print('\nFinal Test Accuracy:')
  train_gen_model(device, sess, model, data['X_test'], data['y_test'], epochs=1,
              batch_size=1, is_training=False, log_freq=100, plot_loss=False)

  # Save Model Checkpoint
  save_model_checkpoint(sess, saver, ckpt_path, prev_epochs + epochs)
