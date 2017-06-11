import shutil

from models.resnet import setup_resnet_2d_model
from utilities.train import *


def run_resnet_test(data, num_classes, device, recover, ckpt_path, prev_epochs,
                    epochs, lr=1e-3, tensorboard_log_dir=None,
                    train_epochs_per_validation=1, dataset=None, tag=None):
    # Create Model
    print("Setting up model...")
    data_shape = list(data['X_train'][0].shape)
    model = setup_resnet_2d_model(data_shape, num_classes, learning_rate=1e-3)
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
                                 "RN-%s-lr-%s-tag-%s-train" % (
                                     dataset, lr, tag))
        val_dir = os.path.join(os.path.expanduser(tensorboard_log_dir),
                               "RN-%s-lr-%s-tag-%s-val" % (
                                   dataset, lr, tag))

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

    for i in range(num_train_val_cycles):
        # Train Model
        print("Training model...")
        train_model(device, sess, model, data['X_train'], data['y_train'],
                    epochs=1,
                    batch_size=64, is_training=True, log_freq=100,
                    plot_loss=False, global_step=global_step,
                    writer=train_writer)

        global_step += train_epochs_per_validation - 1

        # Validate Model
        if tensorboard_log_dir:
            print("Validating model...")
            train_model(device, sess, model, data['X_val'], data['y_val'],
                        epochs=1, batch_size=64, is_training=False,
                        log_freq=100, plot_loss=False, global_step=global_step,
                        writer=val_writer)
        global_step += 1


    # Check Final Training Accuracy
    print("\nFinal Training Accuracy:")
    train_model(device, sess, model, data['X_train'], data['y_train'],
                epochs=1,
                batch_size=64, is_training=False, log_freq=100, plot_loss=False)

    # Check Validation Accuracy
    print('\nFinal Validation Accuracy:')
    train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
                batch_size=64, is_training=False, log_freq=100, plot_loss=False)

    # Save Model Checkpoint
    save_model_checkpoint(sess, saver, ckpt_path, prev_epochs + epochs)
