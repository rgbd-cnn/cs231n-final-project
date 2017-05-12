import os
import re
import fnmatch
from tests.test_resnet_2d import *
from tests.test_inception_resnet_2d import *

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
  print('')

  # Run Appropriate Network
  if network == 'resnet':
    run_resnet_2d_test(device, recover, 'checkpoints/' + model_name, highest_epochs, epochs, debug)
  elif network == 'inception_resnet':
    run_inception_resnet_2d_test(device, recover, 'checkpoints/' + model_name, highest_epochs, epochs, debug)
  else:
    print("Error: Invalid network...")
  
if __name__ == '__main__':
  main()
  exit(0)
