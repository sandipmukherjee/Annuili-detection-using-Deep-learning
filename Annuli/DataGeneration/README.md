# Data Generation

The latest Data set generation script is DataGeneration2.py.


## generate_datasets.py

`generate_datasets.py [opts] [cleaned_data_dir]` generates a pickled file named `dataset.pkl` in the current working directory by default. Change the output file by passing the `--out [file]` option.

Each of the three files contains a single array. On each row is a sample, the last element of each row is the label (0 for "no annulus", 1 for "annulus" or, in the multi-class case, 1 for "left annulus", 2 for "right annulus").

By default, the whole dataset is taken. To restrict the dataset to 2/4 chamber view images, pass `--2ch` or `--4ch`. To further restrict it to left/right annuli, pass `--left` or `--right`.

To generate an unbalanced test set, pass `--unbalanced-test` to the script.

This implementation should be much faster (but will not, at the moment, yield exactly the same dataset) than DataGeneration[2].py by using NumPy arrays instead of lists and helping the garbage collector with Python's `del` keyword.

Sample invocations:

- `python generate_datasets.py --2ch --out dataset-2ch-multiclass-60px-unbal_test.pkl --patch-size 60 --multiclass ~/Desktop/DLsamples_cleaned`
  This takes all 2 chamber images and generates a dataset with 60x60 samples and all three classes (no annulus/left annulus/right annulus)


## DataGeneration2.py

It can be run with the following command:

python DataGeneration2.py path_to_images

It generates 9 annuli image patches of 20 x 20 and then To have more annuli data, the program stores duplicates of 9 patches. Finally, the entire data set is shuffled to get randomized data.


## dataProcessing.py
First data processing implementation. It can be run with the following command:

python dataProcessing.py path_to_images

It generates 20 X 20 patches for annuli and non annuli every 10 pixel of an image. To have more annuli data, the program copies the annuli patch after every 5th non annuli patch.

It stores the 2 chamber and 4 chamber data in the present directory in files named as "tomtec2chamber.pkl" and "tomtec4chamber.pkl" respectively.


## AnnuliDataBuild.py
Data Generation with overlapped images

### Running Instructions

Go inside the directory where the images reside and then Run it with `python AnnuiliDataBuild.py` and it will create a pickle file with the data.

### Specification
All the patches are 40x40 hence has 1600 features and 1 output class(0 for non-annuili or 1 for annuili).
Hence the input size is 1601 with last column as class.

### How the patched are created

I assumed that the annuili resides at the center 20x20 patch of the image.
Created 9 40x40 patches with annuili taking into account the side pixels of the annuili center.
For non-annuli overlapping patches has been taken from other parts of the image apart from the center
