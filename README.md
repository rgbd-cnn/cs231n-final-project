### `cs231n-final-project`
##### Authors: Jakub Dworakowski (jakub@stanford.edu), Tyler Mau (tylermau@stanford.edu), Bohan Wu (bohanwu@stanford.edu)

#### `Logistics:`
1. Remember to create your own dev branch before merging into master remotely: e.g. bohanwu-dev

#### `Dataset List:`
1. [UCL Laundry list](http://www0.cs.ucl.ac.uk/staff/M.Firman/RGBDdatasets/)

#### `Datasets:`
1. [NYU RGB-D Dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* Dataset info
    * Download [nyu_dpth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) first and use `data/NYU-Depth-v2/load_mat.py` to extract images out of it 
    * 1449 (640 x 480) images containing 894 object types: books, bottles, etc.
    * depth info included
* `data/NYU-Depth-v2/load_mat.py`
    * `save_original_images_to_disk()` saves all 1449 images in `./raw-images`
    * `save_single_object_images_to_disk()` creates 894 sub-folders under `./single-object-images` and put all images containing a specific object type in one of these 894 sub-folders
    * `save_cropped_single_object_images_to_disk()` crops all pixels of an image that are labeled as a specific object class using a `rectangle` and save this cropped image to 894 individual sub-folders under `./cropped-single-object-images`. Since these images vary in sizes, we are not yet sure how this could be useful. 
    * more functions to come based on necessity
2. [U-WASH RGB-D Dataset](http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/)
* Analyzing currently.
3. [Princeton the SUN RGB-D Dataset](http://rgbd.cs.princeton.edu/challenge.html)
* Dataset info
    * The Training Set, [SUNRGBD](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip), contains 10355 RGB-D scene images: office, bookstore, etc.
        * Contains images `originally` from:
            1. NYU depth v2 (dimension: `427 x 561`)
            2. Berkeley B3DO (dimension: `427 x 561`)
            3. SUN3D (dimensions: various, such as `530 x 730`, `531 x 681`, `441 x 591`)
    * The Test Set, [SUNRGBDv2Test](http://rgbd.cs.princeton.edu/data/LSUN/SUNRGBDLSUNTest.zip), contains 2860 RGB-D scene images.
* `data/Princeton-SUNRGB-D/load_data.py`
    * `get_rgbd_training_set()` generates X_train, y_train
        * X_train is 10,355 x (various height x various width) x 4. 
            * 10,355 is num training examples.
            * 4 is all channels with the last being depth. 
        * y_train is 10,355 english scene names
    * `get_rgbd_test_set` generates X_test
        * X_test is 2860 x 640 x 480 x 4. 
            * 2860 is num test examples.
            * 640 is image width
            * 480 is image height
            * 4 is all channels with the last being depth

