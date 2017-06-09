import shutil

from tensorflow.contrib.tensorboard.plugins import projector

from models.depth_enhanced_cnn import setup_depth_enhanced_cnn_model
from utilities.train import *


def list_variables(path):
    reader = tf.train.NewCheckpointReader(path)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    result = []
    for name in names:
        result.append((name, variable_map[name]))
    return result


def recover_model(path, sess, ckpt_path, ckptname):
    checkpoint_vars = [tup[0] for tup in list_variables(path)]
    model_vars = [var.name[:var.name.find(':')] for var in
                  tf.all_variables()]

    common_set = set.intersection(set(checkpoint_vars), set(model_vars))
    # print(len(list(common_set)))
    # print(set(checkpoint_vars) - common_set)

    common_checkpoint_vars = {}
    for tup in list_variables(path):
        if tup[0] in common_set:
            common_checkpoint_vars[tup[0]] = tup[1]

    common_model_vars = {}
    model_name_to_vars = {}
    for var in tf.all_variables():
        name = var.name[:var.name.find(':')]
        model_name_to_vars[name] = var
        if name in common_set:
            common_model_vars[name] = var.get_shape().as_list()

    for i in tf.all_variables():
        print(i)

    # print(common_checkpoint_vars)
    # print(common_model_vars)
    print(len(common_checkpoint_vars))
    print(len(common_model_vars))

    valid_vars = []

    for name in common_checkpoint_vars:
        if common_checkpoint_vars[name] != common_model_vars[name]:
            print(name, common_checkpoint_vars[name],
                  common_model_vars[name])
        else:
            valid_vars.append(name)

    print(len([model_name_to_vars[name] for name in valid_vars]))
    saver = tf.train.Saver(
        var_list=[model_name_to_vars[name] for name in valid_vars])
    recover_model_weights(sess, saver, ckpt_path, ckptname)


def tSNE(LOG_DIR):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "final_embedding:0"
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    projector.visualize_embeddings(summary_writer, config)


def run_depth_enhanced_cnn_test(data, num_classes, device, recover, ckpt_path,
                                prev_epochs, epochs, lr=1e-3,
                                train_epochs_per_validation=100,
                                tensorboard_log_dir=None, dataset=None,
                                branch1='IR2d', branch2='IRd', reg=0.0,
                                keep_prob=None, feature_op="stack", tag=None,
                                transfer_learn=None, save_depth=None,
                                visualize_first_layer=None):
    # Create Model
    print("Setting up model...")
    data_shape = list(data['X_train'][0].shape)
    model = setup_depth_enhanced_cnn_model(data_shape, num_classes, 1, 2, 1,
                                           learning_rate=lr, branch1=branch1,
                                           branch2=branch2, reg=reg,
                                           keep_prob=keep_prob,
                                           feature_op=feature_op,
                                           dataset=dataset)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # print(tf.get_collection(tf.GraphKeys.VARIABLES))

    if transfer_learn:
        print("loading depth prediction...")
        model['net'].load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '../models/depth/NYU_ResNet-UpProj.npy'), sess)
        print("done...")

    if recover:
        print("Recovering model...")
        recover_model('checkpoints/27-25', sess, "checkpoints", "27-25")
        # recover_model('tb_checkpoints/9393-5', sess, "tb_checkpoints",
        # "9393-5")

    num_train_val_cycles = epochs / train_epochs_per_validation

    train_log_dir = "DE-%s-%s-%s-lr-%s-reg-%s-prob-%s-FeatOp-%s-tag-%s-train" \
                    % (
                        branch1, branch2, dataset, lr, reg, keep_prob,
                        feature_op,
                        tag)

    val_log_dir = "DE-%s-%s-%s-lr-%s-reg-%s-prob-%s-FeatOp-%s-tag-%s-val" % (
        branch1, branch2, dataset, lr, reg,
        keep_prob, feature_op, tag)

    if tensorboard_log_dir:
        train_dir = os.path.join(os.path.expanduser(tensorboard_log_dir),
                                 train_log_dir)
        val_dir = os.path.join(os.path.expanduser(tensorboard_log_dir),
                               val_log_dir)

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
    best_accuracy = 0
    complete_confusion = []
    for i in range(num_train_val_cycles):
        train_model(device, sess, model, data['X_train'], data['y_train'],
                    epochs=train_epochs_per_validation,
                    batch_size=128, is_training=True, log_freq=100,
                    plot_loss=False, global_step=global_step,
                    writer=train_writer, depth_enhanced=True,
                    X_data_unnormalized=data['X_train_unnormalized'],
                    save_depth=save_depth, log_dir=train_dir, dict=data['dict'])

        global_step += train_epochs_per_validation - 1
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(tensorboard_log_dir, train_log_dir,
                                      'model.ckpt'), i)

        if visualize_first_layer:
            image = data['X_val'][:128]
            label = data['y_val'][:128]
            RGB, D, dmap = sess.run([model['first_layer_b1'],
                                     model['first_layer_b2'],
                                     model['depth_map']],
                                    feed_dict={
                                        model['X']: image,
                                        model['y']: label,
                                        model['is_training']: False,
                                        model['X_unnormalized']:
                                            data['X_val_unnormalized'][:128]
                                    })
            # print(RGB)
            # print(D)
            with open('first_layer.json', 'w') as f:
                json.dump({
                    "rgb": RGB.tolist(),
                    'd': D.tolist(),
                    'dict': data['dict'],
                    'label': data['y_val'][:128].tolist(),
                    'image': data['X_val_unnormalized'][:128].tolist(),
                    'depth': dmap.tolist()
                }, f)

        # Validate Model
        print("\nValidating model...")
        _, accuracy, confusion = train_model(device, sess, model, data['X_val'],
                                             data['y_val'], epochs=1,
                                             batch_size=128, is_training=False,
                                             log_freq=100, plot_loss=False,
                                             global_step=global_step,
                                             writer=val_writer,
                                             depth_enhanced=True,
                                             X_data_unnormalized=data[
                                                 'X_val_unnormalized'],
                                             save_depth=save_depth,
                                             log_dir=val_dir, dict=data['dict'])

        global_step += 1

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(tensorboard_log_dir, val_log_dir,
                                      'model.ckpt'), i)

        # if best_accuracy < accuracy:
        best_accuracy = accuracy
        complete_confusion += confusion
        with open('confusion.json', 'w') as f:
            json.dump({"epoch": i,
                       "accuracy": best_accuracy,
                       "data": complete_confusion,
                       'labels': data['dict']
                       }, f)
        f.close()

    # Check Final Training Accuracy
    print("\nFinal Training Accuracy:")
    train_model(device, sess, model, data['X_train'], data['y_train'], epochs=1,
                batch_size=128, is_training=False, log_freq=100,
                plot_loss=False,
                depth_enhanced=True,
                X_data_unnormalized=data['X_train_unnormalized'],
                save_depth=save_depth)

    # Check Validation Accuracy
    print('\nFinal Validation Accuracy:')
    train_model(device, sess, model, data['X_val'], data['y_val'], epochs=1,
                batch_size=128, is_training=False, log_freq=100,
                plot_loss=False,
                depth_enhanced=True,
                X_data_unnormalized=data['X_val_unnormalized'],
                save_depth=save_depth)

    # Check Test Accuracy
    print('\nFinal Test Accuracy:')
    train_model(device, sess, model, data['X_test'], data['y_test'], epochs=1,
                batch_size=128, is_training=False, log_freq=100,
                plot_loss=False,
                depth_enhanced=True,
                X_data_unnormalized=data['X_test_unnormalized'],
                save_depth=save_depth)

    # Save Model Checkpoint
    save_model_checkpoint(sess, saver, ckpt_path, prev_epochs + epochs)
