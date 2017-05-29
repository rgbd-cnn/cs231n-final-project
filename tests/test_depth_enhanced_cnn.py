import os
import shutil

from models.depth_enhanced_cnn import setup_depth_enhanced_cnn_model
from utilities.train import *


def run_depth_enhanced_cnn_test(data, num_classes, device, recover, ckpt_path,
                                prev_epochs, epochs, lr=1e-3,
                                train_epochs_per_validation=100,
                                tensorboard_log_dir=None, dataset=None,
                                branch1='IR2d', branch2='IRd', reg=0.0,
                                keep_prob=None, feature_op="stack", tag=None):
    # Create Model
    print("Setting up model...")
    data_shape = list(data['X_train'][0].shape)
    model = setup_depth_enhanced_cnn_model(data_shape, num_classes, 1, 2, 1,
                                           learning_rate=lr, branch1=branch1,
                                           branch2=branch2, reg=reg,
                                           keep_prob=keep_prob,
                                           feature_op=feature_op)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Recover Saved Model (if available)
    if recover:
        print("Recovering model...")
        recover_model_checkpoint(sess, saver, 'checkpoints/')

    num_train_val_cycles = epochs / train_epochs_per_validation

    if tensorboard_log_dir:
        train_dir = os.path.join(os.path.expanduser(tensorboard_log_dir),
                                 "DE-%s-%s-%s-lr-%s-reg-%s-prob-%s-FeatOp-%s-tag-%s-train" % (
                                     branch1, branch2, dataset, lr, reg,
                                     keep_prob, feature_op, tag))
        val_dir = os.path.join(os.path.expanduser(tensorboard_log_dir),
                               "DE-%s-%s-%s-lr-%s-reg-%s-prob-%s-FeatOp-%s-tag-%s-val" % (
                                   branch1, branch2, dataset, lr, reg,
                                   keep_prob, feature_op, tag))

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

    # Train Model
    print("Training model...")
    for i in range(num_train_val_cycles):
        train_model(device, sess, model, data['X_train'], data['y_train'],
                    epochs=train_epochs_per_validation,
                    batch_size=64, is_training=True, log_freq=100,
                    plot_loss=False, global_step=global_step,
                    writer=train_writer, depth_enhanced=True)

        global_step += train_epochs_per_validation - 1

        # Validate Model
        print("\nValidating model...")
        train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
                    batch_size=64, is_training=False,
                    log_freq=100, plot_loss=False, global_step=global_step,
                    writer=val_writer, depth_enhanced=True)
        print('')
        global_step += 1

    # Check Final Training Accuracy
    print("\nFinal Training Accuracy:")
    train_model(device, sess, model, data['X_train'], data['y_train'], epochs=1,
                batch_size=64, is_training=False, log_freq=100, plot_loss=False,
                depth_enhanced=True)

    # Check Validation Accuracy
    print('\nFinal Validation Accuracy:')
    train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
                batch_size=64, is_training=False, log_freq=100, plot_loss=False,
                depth_enhanced=True)

    # Check Test Accuracy
    print('\nFinal Test Accuracy:')
    train_model(device, sess, model, data['X_test'], data['y_test'], epochs=1,
                batch_size=64, is_training=False, log_freq=100, plot_loss=False,
                depth_enhanced=True)

    # Save Model Checkpoint
    save_model_checkpoint(sess, saver, ckpt_path, prev_epochs + epochs)
