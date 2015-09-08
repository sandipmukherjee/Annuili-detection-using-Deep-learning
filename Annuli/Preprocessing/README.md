This code cleans the ultrasound lines and other artifacts from the images and converts them to grayscale.

Run it with `python preprocessing.py` in a directory with jpeg images.
If you don't want to replace the images with their cleaned versions, do `python preprocessing.py --no-replace` instead.

Performance is not optimal (~ 30 seconds for 5 images for on my computer), but the code only has to be run once on the whole dataset.
If you have multiple cores, you can easily parallelize the operation by moving different parts of the dataset to different directories. Then run the script from the commandline in each of the directories separately, but at the same time.