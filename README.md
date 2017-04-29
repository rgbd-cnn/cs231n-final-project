### `cs231n-final-project`
##### Authors: Jakub Dworakowski (jakub@stanford.edu), Tyler Mau (tylermau@stanford.edu), Bohan Wu (bohanwu@stanford.edu)

#### `Logistics:`
1. Remember to create your own dev branch before merging into master remotely: e.g. bohanwu-dev

#### `Dataset List:`
1. [UCL Laundry list](http://www0.cs.ucl.ac.uk/staff/M.Firman/RGBDdatasets/)

#### `Datasets:`
1. [NYU RGB-D Dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* Dataset info
    * 1449 (640 x 480) images containing 894 object types: books, bottles, etc.
    * depth info included
* Image extraction python script in `data/nyu_depth_v2/load_mat.py`
    * `save_original_images_to_disk()` saves all 1449 images in `raw-images`
    * `save_single_object_images_to_disk()` creates 894 subfolders under `single-object-images` and put all images containing a specific object type in one of these 894 subfolders
    * more functions to come based on necesscity
2. [U-WASH RGB-D Dataset](http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/)
* Analyzing currently.
3. [Princeton the SUN RGB-D Dataset](http://rgbd.cs.princeton.edu/challenge.html)
* Analyzing currently.

