### `cs231n-final-project`
##### Authors: Jakub Dworakowski (jakub@stanford.edu), Tyler Mau (tylermau@stanford.edu), Bohan Wu (bohanwu@stanford.edu)

#### `Logistics:`
1. Remember to create your own dev branch before merging into master remotely: e.g. bohanwu-dev

#### `Dataset List:`
1. [UCL Laundry list](http://www0.cs.ucl.ac.uk/staff/M.Firman/RGBDdatasets/)

#### `Datasets:`
1. `SCENES ONLY` [NYU RGB-D Dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 
    * Dataset info
        * Download [nyu_dpth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) first and use `data/NYU-Depth-v2/load_mat.py` to extract images out of it 
        * 1449 (640 x 480) images containing 894 object types: books, bottles, etc.
        * depth info included
    * `data/NYU-Depth-v2/load_mat.py`
        * `save_original_images_to_disk()` saves all 1449 images in `./raw-images`
        * `save_single_object_images_to_disk()` creates 894 sub-folders under `./single-object-images` and put all images containing a specific object type in one of these 894 sub-folders
        * `save_cropped_single_object_images_to_disk()` crops all pixels of an image that are labeled as a specific object class using a `rectangle` and save this cropped image to 894 individual sub-folders under `./cropped-single-object-images`. Since these images vary in sizes, we are not yet sure how this could be useful. 
        * more functions to come based on necessity
2. `SINGLE OBJECTS` [U-WASH RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/) 
    * Dataset info
        * [University of Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar)
        * Contains 30 household single object classes like apple, towels, etc.
            * these images have different heights and widths and need to be resized
    * `data/UWASH-RGBD/load_data.py`
        * `get_np_arrays_from_dataset(data_dir, height, width, save)` generates:
            * X: np.array of shape (?, H, W, 4), where the 4th channel is depth
            * Y: list of label indices, 0, 1, 2, .... etc.
            * labels: dict where key = label index, value = label name (apple, banana, etc.) 
        * Parameters:
            * data_dir: `str` Directory of the UWash dataset
            * height: `int` arbitrary resized image height
            * width: `int` arbitrary resized image width
            * save: `bool` whether to save the new, resized image to disk
                * if save is `True`, then new folders with suffix "_resized" are created with the same file structures as the original dataset folders.
                * e.g. resized rgb file saved to: `rgbd-dataset/apple_resized/apple_1/apple_1_1_1_crop.png`
                * e.g. resized depth file saved to `rgbd-dataset/apple_resized/apple_1/apple_1_1_1_depthcrop.png`
            * overwrite: `bool` whether to overwrite pickles are already exist in the directory
        * Number of training examples: 600k, number of classes: 51-ish
        * Script use horizontal flip and vertical flip for data augmentation
          
3. `SCENES ONLY` [Princeton the SUN RGB-D Dataset](http://rgbd.cs.princeton.edu/challenge.html)
    * Dataset info
        * The Training Set, [SUNRGBD](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip), contains 10355 RGB-D scene images: office, bookstore, etc.
            * Contains images `originally` from:
                1. NYU depth v2 (dimension: `427 x 561`)
                2. Berkeley B3DO (dimension: `427 x 561`)
                3. SUN3D (dimensions: various, such as `530 x 730`, `531 x 681`, `441 x 591`)
        * The Test Set, [SUNRGBDv2Test](http://rgbd.cs.princeton.edu/data/LSUN/SUNRGBDLSUNTest.zip), contains 2860 RGB-D scene images.
    * `data/Princeton-SUNRGB-D/load_data.py`
        * `get_rgbd_training_set()` generates X_train (width, height resized to `640 x 480`), y_train
            * X_train is 10,355 x 480 x 640 x 4. 
                * 10,355 is num training examples.
                * 640 is image width
                * 480 is image height
                * 4 is all channels with the last being depth. 
            * y_train is 10,355 english scene names
        * `get_rgbd_test_set` generates X_test
            * X_test is 2860 x 480 x 640 x 4. 
                * 2860 is num test examples.
                * 640 is image width
                * 480 is image height
                * 4 is all channels with the last being depth

#### `Relevant Works Cited` 
1. [RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features](https://pdfs.semanticscholar.org/efa3/e8826aab1a79d05b1f3ab55b277c0120a092.pdf)
2. [FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture.](http://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf)
3. [Convolutional-Recursive Deep Learning for 3D Object Classification](https://papers.nips.cc/paper/4773-convolutional-recursive-deep-learning-for-3d-object-classification.pdf)

     