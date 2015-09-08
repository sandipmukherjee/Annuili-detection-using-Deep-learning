Wafer
=====

TODO
----
1. Look at data
2. If necessary write code to clean up the dataset
3. Write code to read the dataset, in batches, in a NumPy/Theano-processable way
4. Write preprocessing (normalization to mean 0, divide by stddev, scale to [0,1], etc.) if needed
5. Implement, understand & run deeplearning.net convolutional NN implementation, or use something from PyLearn2.
6. Create the training data, including consistent training/validation/test set split (60/20/20)
7. Set up training infrastructure
8. Check that the net learns *something*
9. Visualize what the net learned by mapping an input image to a "heat map" using the output.
10. Think about (deep) architecture, including use of autoencoders for pretraining and whatever else G. Hinton mentioned
11. Deal with things like class imbalance (throw away some "good" patches or repeat some "bad" patches)
12. If necessary, increase training set size by mirroring and the like; optimize training speed; 

Looking at the Data
-------------------
- Usually regular structure at small level
- A whole batch being bad is relatively unlikely, but not impossible
- Simple approach is to compare blurred images with the mean
- There is noise, but that can mostly be ignored for now
- Errors are
	- disturbances in the micro-structure by dust particles and whatever causes "gashes". However, the micro-structure is not always visible in good chips. A disturbance does not mean "no microstructure visible", but "different from the environment".
	- regions that are somewhat darker than they "should" be. However, some natural variance in brightness always occurs and is ok (sometimes, pretty strong such variance is ok)
	- Some chips are "bad" as a whole. I'm not able to tell why.
	- Cases where some liquid did not reach the edge. Often, the whole batch is bad in these cases
	- Sometimes, the difference is only in a color tint. These cases are probably where some liquid was empty. This probably accounts for all cases where chips (and batches) are bad as a whole
	- dust particles don't have to actually disturb the microstructure, but are sometimes just much lighter than the environment.

- Presumably, autoencoders whould learn to recognize the microstructure and to ignore weird color stuff.
- Sometimes, it is actually easier to see errors in a (probably bicubic) downsampled image
- It's best to ignore the outer edge
- We obviously can't count on all images being in the right category. Just look at the "good" images of ASL3C-8-1LVC1G08EIM. Some of these are obviously faulty.
- Removing a (rotation-corrected) batch mean (without a lot of blurring) is a good idea. Just look at e.g BICOM3-1THS7327A1IMX.
- Might a good architecture actually take 2 input images, namely the batch mean and the test image? (First few layers for feature extraction of each one separately, then one that compares the two images.) => When looking at the dataset, it becomes clear that even a human expert can detect some (but not all) errors only by looking at the whole batch. Hence, the network has to take all images in the batch into account somehow.
- the dataset is probably not sufficient to handle cases where the whole batch is bad properly
- probably use median instead of mean so as not to move the image we use for difference by the errors
- Instead of doing things like difference between batch mean/median and individual images yourself, let the data decide. Generate each preprocessing output pixel value by combining the 5x5 environment of that pixel in both the current image and the batch mean/median with each other. This one is the same for all pixels. Since we think that difference is probably right, just reflect that in the initialization, i.e 00000;00000;00100;00000;0000; +- N(0,0.1), but non-zero. This would have to be part of normal (gradient descent) training. Maybe even do the "preprocessing" sub-network without a separate hidden layer.
- If images are scaled/normalized, they should probably not be scaled wrt their own mean/stddev, but that of the batch mean
- The preprocessing step might be subsumed by a normal convolutional NN architecture that takes as input the normal image and the batch median. There, we would have local connectivity for the original image itself (as in the deeplearning tutorial) and add the corresponding pixels of the batch median as local inputs as well.
- Single images with that are "obviously" wrong should also be recognized as such (for a case, see LBC7X-1TPS51219B1/1240)

Dependencies
------------
- Python with NumPy and PIL
- ImageMagick >= 6.6.6
- Torch7, with the 'csv', 'fs', 'image' and 'nn' packages, all of which can be installed using `torch-pkg install [package]`
  If there are problems, try to install Torch with `torch-pkg install torch WITH_LUA_JIT=0`.

Preprocessing
-------------
1. Rotate all images so that they are as aligned as possible
2. Remove 10 px around the corner of each image as the corners are likely to lead to problems due to their "untypical" look
3. Compute the median of all the (rotated,cropped) images using the `convert` command of ImageMagick.
   We use the median because the mean, while faster, would be distorted by errors in the batch, whereas the median is mostly immune from this.
  
Preprocessing is invoked by running preprocessing.py with the directory where the dataset is located as 1st argument, the directory to which cleaned files should be written as 2nd argument. For now, it assumes that the masks of error pixels are located in `[dataset]/../masks TI`.

The results of preprocessing are stored in disk, as JPG files. We do this instead of storing each batch in a .pkl file because of disk and memory usage.

Training set
------------
The data is split into training/validation/test set using a 60/20/20 split on the batches.
The amount of training data is increased using horizontal and vertical mirroring.

What do we want to learn?
-------------------------
For starters, we only want to recognize local errors, i.e we ignore the case where a chip as a whole is faulty.

Network Architecture
--------------------
The architecture is basically a CNN, but with a few additions:
- A basic idea is to somehow bringing the median of each batch (*) into the network for each individual image in the batch. A simple approach for this is to subtract the median image. However, there is no reason not to let the network learn what the relationship between each image and the batch median should be.
  Hence, the input not only consists of the input image, but also of the batch median.
- Layers:
  (0. Mean + Std.dev removal (probably). Not that important if images are all in a similar range)
  1. 5x5 convolutional layer for both the input image and the median separately (but with shared weights)
  2. Combine the image and the batch median by a 2x3x3 convolutional layer, i.e it takes 3x3 inputs from both the image and batch mean layers
  3. Some more convolutional layers
  4. A fully connected MLP layer with a tanh activation function
  5. A (logistic regression) sigmoid output.
  
(*): Take all images in the batch, and take, for each pixel, the mean of the corresponding values in each image.
  The rationale behind this is that some of the error are very hard to spot without the other images in the batch as reference.

Training
--------
Training is done using gradient descent with rmsprop (and possibly momentum). 