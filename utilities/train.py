import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import tensorflow as tf
import tensorlayer
import matplotlib.colors as colors

# Taken from:
# https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

def shiftedColorMap(cmap, start=-3.0, midpoint=0, stop=3.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 1024)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 512, endpoint=False),
        np.linspace(midpoint, 1.0, 512, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

# Save Checkpoint of Model
def save_model_checkpoint(session, saver, filename, epoch_num):
  save_path = saver.save(session, filename, epoch_num)
  print("\nCheckpoint saved in file: %s" % save_path)


# Recover Saved Model Checkpoint
def recover_model_checkpoint(session, saver, checkpoint_path):
  saver.restore(session, tf.train.latest_checkpoint(checkpoint_path))
  print("Model restored!\n")


def train_gen_model(device, sess, model, X_data, labels, epochs=1, batch_size=1, is_training=False,
                    log_freq=100, plot_loss=False, global_step=None, writer=None):
  norm = colors.Normalize(vmin=-1.,vmax=1.)
  # shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shifted')
  with tf.device(device):
    # Calculate Prediction Accuracy
    y = tf.image.resize_images(model['y'], [32, 32],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mse = tf.reduce_mean(tf.squared_difference(model['y_out'], y))

    # Shuffle Training Data
    train_indicies = np.arange(X_data.shape[0])
    np.random.shuffle(train_indicies)

    # Populate TensorFlow Variables
    variables = [tf.reduce_mean(model['loss_val']), model['y_out'], model['y_res'], mse]
    if is_training:
      variables[-1] = model['train_step']

    # Iteration Counter
    iter_cnt = 0
    losses = []
    print(epochs)
    for epoch in range(epochs):
      print(epoch)
      # Keep Track of Loss and Number of Correct Predictions
      num_correct = 0
      epoch_loss = 0

      # Iterate Over the Entire Dataset Once
      for i in range(int(math.ceil(X_data.shape[0] / float(batch_size))) - 1):
        # Indices for Batch
        start_idx = (i * batch_size) % X_data.shape[0]
        idx = train_indicies[start_idx:start_idx + batch_size]
        actual_batch_size = labels[i:i + batch_size].shape[0]

        # Feed Dictionary for Batch
        feed_dict = {model['X']: X_data[idx, :],
                     model['y']: labels[idx, :],
                     model['is_training']: is_training}
        # # 'Label' for image

        # Run TF Session (Returns Loss and Correct Predictions)
        loss, img, yinp, _ = sess.run(variables, feed_dict=feed_dict)

        if (np.sum(img) == 0):
          print "WARNING: predicted image came back as blank!!!!"

        num, _, _, _ = X_data[idx].shape
        for x in range(num):
          # Image to be estimated
          # Image estimated
          fig = plt.figure(0)
          plt.imshow(X_data[idx][x, :, :, :]/256)
          plt.show(block=False)
          fig = plt.figure(1)
          ii = plt.imshow(yinp[x, :, :, 0], interpolation='nearest', cmap=matplotlib.cm.get_cmap('viridis'))
          plt.colorbar(ii)
          plt.show(block=False)
          fig = plt.figure(2)
          ii = plt.imshow(yinp[x, :, :, 0] - img[x, :, :, 0], interpolation='nearest', cmap=matplotlib.cm.get_cmap('bwr'))
          plt.colorbar(ii)
          plt.show(block=False)
          fig = plt.figure(3)
          ii = plt.imshow(img[x, :, :, 0], interpolation='nearest', cmap=matplotlib.cm.get_cmap('viridis'))
          plt.colorbar(ii)
          plt.show()

        # print(loss)
        # num_correct += np.sum(corr)
        epoch_loss += loss * actual_batch_size

        # # Print Loss and Accuracies
        if is_training and (iter_cnt % log_freq) == 0:
          print("Iteration = ", iter_cnt, " Training Loss ", loss)
        iter_cnt += 1

      # Calculate Performance
      # accuracy = num_correct / float(X_data.shape[0])
      total_loss = epoch_loss / float(X_data.shape[0])
      losses.append(total_loss)
      print("Epoch = ", epoch, " Overall Loss ", total_loss)

      if writer is not None:
        global_step += 1
        summary = tf.Summary()
        # summary.value.add(tag="Accuracy", simple_value=accuracy)
        summary.value.add(tag="Loss", simple_value=total_loss)
        writer.add_summary(summary, global_step=global_step)

    if plot_loss:
      plt.plot(losses)
      plt.grid(True)
      plt.xlim(-10, 800)
      plt.title('Total Loss vs. Epoch')
      plt.xlabel('Epoch')
      plt.ylabel('Total Loss')
      plt.show()

  return total_loss

# Train the Model
def train_model(device, sess, model, X_data, labels, epochs=1, batch_size=64, is_training=False, log_freq=100,
                plot_loss=False, global_step=None, writer=None):
  with tf.device(device):
    # Calculate Prediction Accuracy
    correct_prediction = tf.equal(tf.argmax(model['y_out'], 1), model['y'])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Shuffle Training Data
    train_indicies = np.arange(X_data.shape[0])
    np.random.shuffle(train_indicies)

    # Populate TensorFlow Variables
    variables = [model['loss_val'], correct_prediction, accuracy]
    if is_training:
      variables[-1] = model['train_step']

    # Iteration Counter
    iter_cnt = 0
    losses = []
    for epoch in range(epochs):
      # Keep Track of Loss and Number of Correct Predictions
      num_correct = 0
      epoch_loss = 0

      # Iterate Over the Entire Dataset Once
      for i in range(int(math.ceil(X_data.shape[0] / float(batch_size)))):
        # Indices for Batch
        start_idx = (i * batch_size) % X_data.shape[0]
        idx = train_indicies[start_idx:start_idx + batch_size]
        actual_batch_size = labels[i:i + batch_size].shape[0]

        # Feed Dictionary for Batch
        feed_dict = {model['X']: X_data[idx, :],
                     model['y']: labels[idx],
                     model['is_training']: is_training}

        # Run TF Session (Returns Loss and Correct Predictions)
        loss, corr, _ = sess.run(variables, feed_dict=feed_dict)
        # print(loss)
        num_correct += np.sum(corr)
        epoch_loss += loss * actual_batch_size

        # Print Loss and Accuracies
        if is_training and (iter_cnt % log_freq) == 0:
          print("Iteration {0}: Training Loss = {1:.3g} and Accuracy = {2:.2g}" \
                .format(iter_cnt, loss, np.sum(corr) / float(actual_batch_size)))
        iter_cnt += 1

      # Calculate Performance
      accuracy = num_correct / float(X_data.shape[0])
      total_loss = epoch_loss / float(X_data.shape[0])
      losses.append(total_loss)

      print("Epoch {0}: Overall Loss = {1:.3g} and Accuracy = {2:.3g}" \
            .format(epoch + 1, total_loss, accuracy))

      if writer is not None:
        global_step += 1
        summary = tf.Summary()
        summary.value.add(tag="Accuracy", simple_value=accuracy)
        summary.value.add(tag="Loss", simple_value=total_loss)
        writer.add_summary(summary, global_step=global_step)

    if plot_loss:
      plt.plot(losses)
      plt.grid(True)
      plt.xlim(-10, 800)
      plt.title('Total Loss vs. Epoch')
      plt.xlabel('Epoch')
      plt.ylabel('Total Loss')
      plt.show()

  return total_loss, accuracy
