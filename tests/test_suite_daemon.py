import argparse
import fnmatch
import re
import sys

from tests.test_inception_resnet import *
from tests.test_resnet import *

from data.cs231n.data_utils import get_CIFAR10_data
from data.uwash_rgbd.load_pickles import load_uwash_rgbd
from tests.test_two_branch_cnn import *


# run example: "python -m tests.test_suite_daemon --network inception_resnet
# --dataset uwash_3d --load False --model_name 1 --epochs 100 --device /gpu:0
#  --debug False 2>error.txt > log.txt"

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type=str, help='network to test',
                        default='inception_resnet')
    parser.add_argument('--dataset', type=str, help='dataset to test',
                        default='uwash_3d')
    parser.add_argument('--load', type=int,
                        help='whether to load existing model', default=0)
    parser.add_argument('--model_name', type=str, help='model name',
                        default=None)
    parser.add_argument('--epochs', type=int,
                        help='number of epochs', default=None)
    parser.add_argument('--device', type=str,
                        help='CPU or GPU', default='/cpu:0')
    parser.add_argument('--debug', type=int,
                        help='debug mode', default=0)
    parser.add_argument('--tensorboard_log_dir', type=str,
                        help='Where to save Tensorboard Logs for '
                             'visualization, setting it to None means disable '
                             'Tensorboard',
                        default=None)
    parser.add_argument('--lr', type=float,
                        help='learning rate', default=1e-3)
    parser.add_argument('--dropout_keep_prob', type=float,
                        help='dropout keep probability', default=1.0)
    parser.add_argument('--reg', type=float,
                        help='regularization', default=0.0)
    parser.add_argument('--train_epochs_per_validation', type=int,
                        help='How many epochs to train before validating once',
                        default=1)
    parser.add_argument('--branch1', type=str,
                        help='model of the first branch', default="IR2d")
    parser.add_argument('--branch2', type=str,
                        help='model of the second branch', default="IR3d")
    parser.add_argument('--feature_op', type=str,
                        help='how to fuse two features', default="stack")
    return parser.parse_args(argv)


def main(args):
    # Suppress Annoying TensorFlow Logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Specify Network
    network = args.network

    # Specify Dataset
    dataset = args.dataset

    # Choose to Load Checkpoint
    load = args.load

    # Specify Model Name
    model_name = args.model_name

    # Find Checkpoint
    recover = False
    highest_epochs = 0
    for file in os.listdir('checkpoints'):
        if fnmatch.fnmatch(file, model_name + '*.data*'):
            recover = True
            num_epochs = int(re.split('\W+', file)[1])

            if num_epochs > highest_epochs:
                highest_epochs = num_epochs
                # most_recent_file = file

    if load and not recover:
        print("Checkpoint not found. Creating new model...")
    if recover and not load:
        print("Checkpoint already exists...")
        exit(-1);

    # Specify Number of Epochs
    epochs = args.epochs

    # Specify Device Type (CPU or GPU)
    device = args.device

    # Specify Debug Mode
    debug = args.debug

    print("\nLoading data...")
    # Get Appropriate Data
    if dataset == 'cifar':
        # Get CIFAR-10 Dataset
        data = get_CIFAR10_data(num_training=49000, num_validation=1000,
                                num_test=1000,
                                subtract_mean=True)
        num_classes = 10
    elif dataset == 'uwash_2d':
        # Get UWASH Dataset (Without Depth)
        data = load_uwash_rgbd(depth=False)
        num_classes = 51
    elif dataset == 'uwash_3d':
        # Get UWASH Dataset (With Depth)
        data = load_uwash_rgbd(depth=True)
        num_classes = 51
    else:
        print("Error: Invalid dataset...")
        exit(-1)

    # Debug Mode: Run on Smaller Dataset
    if debug:
        data['X_train'] = data['X_train'][0:500]
        data['y_train'] = data['y_train'][0:500]
        data['X_val'] = data['X_val'][0:500]
        data['y_val'] = data['y_val'][0:500]

    print("Finished loading data...")
    print("   Training Size:   %d" % data['y_train'].shape[0])
    print("   Validation Size: %d" % data['y_val'].shape[0])
    print("   Test Size:       %d" % data['y_test'].shape[0])
    print('')

    # Run Appropriate Network
    if network == 'resnet':
        run_resnet_test(data, num_classes, device, recover,
                           'checkpoints/' + model_name, highest_epochs, epochs)
    elif network == 'inception_resnet':
        run_inception_resnet_test(data, num_classes, device, recover,
                                     'checkpoints/' + model_name,
                                     highest_epochs,
                                     epochs, lr=args.lr,
                                     train_epochs_per_validation=args.train_epochs_per_validation,
                                     tensorboard_log_dir=args.tensorboard_log_dir,
                                     dataset=dataset, reg=args.reg,
                                     keep_prob=args.dropout_keep_prob)

    elif network == 'two_branch':
        run_two_branch_cnn_test(data, num_classes, device, recover,
                                'checkpoints/' + model_name, highest_epochs,
                                epochs, lr=args.lr,
                                train_epochs_per_validation=args.train_epochs_per_validation,
                                tensorboard_log_dir=args.tensorboard_log_dir,
                                dataset=dataset, branch1=args.branch1,
                                branch2=args.branch2, reg=args.reg,
                                keep_prob=args.dropout_keep_prob, feature_op=args.feature_op)

    else:
        print("Error: Invalid network...")
        exit(-1)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
    exit(0)
