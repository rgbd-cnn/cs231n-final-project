import os
import re
import fnmatch
from tests.test_resnet_2d import *
from tests.test_inception_resnet_2d import *
from data.cs231n.data_utils import get_CIFAR10_data
from data.uwash_rgbd.load_pickles import load_uwash_rgbd

def main():
  # Suppress Annoying TensorFlow Logs
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  print("Welcome to the CNN Test Suite!")

  # Specify Network
  ask = True
  while ask:
    network = input("\nWhich network would you like to test?\n" +
                    "   1. Standard ResNet\n" +
                    "   2. Inception-ResNet\n" +
                    "Please select number: ")
    ask = False
    if network == 1:
      network = 'resnet'
    elif network == 2:
      network = 'inception_resnet'
    else:
      print("Invalid choice...")
      ask = True

  # Specify Dataset
  ask = True
  while ask:
    dataset = input("\nWhich dataset would you like to use?\n" +
                    "   1. CIFAR-10 (2D)\n" +
                    "   2. UWASH (2D)\n" +
                    "   3. UWASH (3D)\n" +
                    "Please select number: ")
    ask = False
    if dataset == 1:
      dataset = 'cifar'
    elif dataset == 2:
      dataset = 'uwash_2d'
    elif dataset == 3:
      dataset = 'uwash_3d'
    else:
      print("Invalid choice...")
      ask = True

  # Choose to Load Checkpoint
  ask = True
  while ask:
    load = input("\nWould you like to load a model checkpoint?\n" +
                 "   1. Create New Model\n" +
                 "   2. Load Saved Model\n" +
                 "Please select number: ")
    ask = False
    if load == 1:
      load = False
    elif load == 2:
      load = True
    else:
      print("Invalid choice...")
      ask = True

  # Specify Model Name
  model_name = raw_input("\nPlease specify model name: ")

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
  epochs = input("\nNumber of Epochs: ")

  # Specify Device Type (CPU or GPU)
  ask = True
  while ask:
    device = input("\nWould you like to run on a CPU or GPU?\n" +
                   "   1. CPU\n" +
                   "   2. GPU\n" +
                   "Please select number: ")
    ask = False
    if device == 1:
      device = '/cpu:0'
    elif device == 2:
      device = '/gpu:0'
    else:
      print("Invalid choice...")
      ask = True

  # Specify Debug Mode
  ask = True
  while ask:
    debug = input("\nWould you like to run in debug mode?\n" +
                  "   1. Normal\n" +
                  "   2. Debug\n" +
                  "Please select number: ")
    ask = False
    if debug == 1:
      debug = False
    elif debug == 2:
      debug = True
    else:
      print("Invalid choice...")
      ask = True

  print("\nLoading data...")
  # Get Appropriate Data
  if dataset == 'cifar':
    # Get CIFAR-10 Dataset
    data = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
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
    run_resnet_2d_test(data, num_classes, device, recover, 'checkpoints/' + model_name, highest_epochs, epochs)
  elif network == 'inception_resnet':
    run_inception_resnet_2d_test(data, num_classes, device, recover, 'checkpoints/' + model_name, highest_epochs, epochs)
  else:
    print("Error: Invalid network...")
    exit(-1)

if __name__ == '__main__':
  main()
  exit(0)
