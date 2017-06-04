import math
import os

import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow as tf


# Save Checkpoint of Model
def save_model_checkpoint(session, saver, filename, epoch_num):
    save_path = saver.save(session, filename, epoch_num)
    print("\nCheckpoint saved in file: %s" % save_path)


# Recover Saved Model Checkpoint
def recover_model_checkpoint(session, saver, checkpoint_path):
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_path))
    print("Model restored!\n")

    # Recover Saved Model Checkpoint


def recover_model_weights(session, saver, checkpoint_path):
    saver.restore(session,
                  os.path.join(checkpoint_path, "27-25"))
    print("Model restored!\n")


def train_gen_model(device, sess, model, X_data, labels, epochs=1,
                    batch_size=64, is_training=False,
                    log_freq=100, plot_loss=False, global_step=None,
                    writer=None):
    with tf.device(device):
        # Calculate Prediction Accuracy
        y = tf.image.resize_images(model['y'], [32, 32],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mse = tf.reduce_mean(tf.squared_difference(model['y_out'], y))

        # Shuffle Training Data
        train_indicies = np.arange(X_data.shape[0])
        np.random.shuffle(train_indicies)

        # Populate TensorFlow Variables
        variables = [tf.reduce_mean(model['loss_val']), mse]
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
                loss, mse = sess.run(variables, feed_dict=feed_dict)
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


def save_depth_maps(X, depth_maps, y_labels, suffix):
    if "depth_maps" not in os.listdir('./'):
        os.mkdir('./depth_maps')
    with open(os.path.join('./depth_maps', suffix + '.json'), 'w') as fp:
        json.dump({'data': depth_maps.tolist(), 'label': y_labels.tolist(), 'X': X.tolist()}, fp)



# Train the Model
def train_model(device, sess, model, X_data, labels, epochs=1, batch_size=64,
                is_training=False, log_freq=100,
                plot_loss=False, global_step=None, writer=None,
                depth_enhanced=False, X_data_unnormalized=None, save_depth=None):
    with tf.device(device):
        # Calculate Prediction Accuracy
        if depth_enhanced:
            num_train = X_data.shape[0]
            num_train = num_train / batch_size * batch_size
            X_data = X_data[:num_train]
            X_data_unnormalized = X_data_unnormalized[:num_train]
            labels = labels[:num_train]

        prediction = tf.argmax(model['y_out'], 1)
        correct_prediction = tf.equal(prediction, model['y'])
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Shuffle Training Data
        train_indicies = np.arange(X_data.shape[0])
        np.random.shuffle(train_indicies)

        # Populate TensorFlow Variables
        if X_data_unnormalized == None or not save_depth:
            variables = [model['loss_val'], correct_prediction, accuracy, prediction, model['y']]
        else:
            variables = [model['loss_val'], model["depth_map"], correct_prediction, accuracy, prediction, model['y']]
        if is_training:
            variables[-1] = model['train_step']

        # Iteration Counter
        iter_cnt = 0
        losses = []
        confusion = []
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
                if X_data_unnormalized == None:
                    feed_dict = {model['X']: X_data[idx, :],
                                 model['y']: labels[idx],
                                 model['is_training']: is_training}
                else:
                    feed_dict = {model['X']: X_data[idx, :],
                                 model['X_unnormalized']: X_data_unnormalized[idx, :],
                                 model['y']: labels[idx],
                                 model['is_training']: is_training}

                # Run TF Session (Returns Loss and Correct Predictions)
                if X_data_unnormalized == None or not save_depth:
                    loss, corr, _, pred, gt = sess.run(variables, feed_dict=feed_dict)
                else:
                    loss, depth_map, corr, _, pred, gt = sess.run(variables, feed_dict=feed_dict)
                    save_depth_maps(X_data_unnormalized[idx, :], depth_map, labels[idx], str(epoch) + "-" + str(i))
                # print(loss)
                num_correct += np.sum(corr)
                epoch_loss += loss * actual_batch_size

                # Print Loss and Accuracies
                if is_training and (iter_cnt % log_freq) == 0:
                    print(
                        "Iteration {0}: Training Loss = {1:.3g} and Accuracy "
                        "= {2:.2g}" \
                        .format(iter_cnt, loss,
                                np.sum(corr) / float(actual_batch_size)))
                iter_cnt += 1

                if not is_training:
                    for i in range(len(gt)):
                        confusion.append((pred[i], gt[i]))

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

    return total_loss, accuracy, confusion
