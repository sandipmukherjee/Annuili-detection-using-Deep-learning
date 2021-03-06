# Annuli data experiments so far

For the annuli data set, we started with the cleaned images generated by preprocessing.py.

Data Generation experiments:

We treated 2 chamber and 4 chamber annuli data as different data sets.

For both data sets, we did the following:

1. First of all, we took 20 x 20 patch from center and replicated it 20 times, 
20 x 20 non overlapping non-annuli patches from the rest of the image apart from the center.

  With this data, we first tried a standard neural network, code for which is in repository called neuralnet.py. The results are as follows:

  ````
  17% validation and test error for 2 chamber data after 1st training epoch, but no significant improvement after that.
  ````

  Then we moved to convolutional neural network, code for which can be found in the repository as ConvolutionalNN.py. The results are as follows:

  ````
  Training 60%, Validation 20% and testing 20% Split
  ````
  
  - 4 chamber data set:
  
    ````
    Best validation score of 3.924138 % obtained at iteration 5484,with test performance 4.824119 %
    ````
    
  - 2 chamber data set:
    
    ````
    Best validation score of 6.897297 % obtained at iteration 10892,with test performance 10.631441 %
    ````


2. Then we decided to balance the annuli and non annuli patches in number, so we created 20 x 20 non annuli patches from the entire image with a moving box every 10 pixel, apart from the center region. 

  For the annuli patches, we took 9 20 x 20 overlapping patches from the center region and to get more annuli data, we duplicated the patches 20 times for 2 chamber data and 10 times for 4 chamber data (less duplication for 4 chamber data due to computation power issue, My computer hanged if I tried 20 times for 4 chamber.)

  The results are as follows:

  ````
  Training 60%, Validation 20% and testing 20% Split (All the data are balanced here (even validation and testing))
  ````
  
  - 4 chamber:

    ````
    epoch 42, minibatch 1946/1946, validation error 2.414506 %
    epoch 42, minibatch 1946/1946, test error of best model 2.428028 %
    ````
  
  - 2 Chamber:
    
    ````
    Best validation score of 2.741667 % obtained at iteration 194544,with test performance 2.808497 %
    ````

3. We decided to balance training and validation split but not the test set. With this approach the results are as follows:

   - 2 chamber
     
     ````
     epoch 3, minibatch 1112/1112, validation error 4.413514 %
     epoch 3, minibatch 1112/1112, test error of best model 5.733333 %
     ````

     IMPORTANT: with this approach, the validation error dropped till 0.23% but after some epochs the training error started increasing and went upto 14% (Detailed logs for this observation can be seen in the `../Logs/2ch_20px_no_rmsprop_lrate_0_1_balanced_testset.log)

4. Due to the increased test set error, we decided to do everything(training, validation and testing) with unbalanced data. It is still in progress.
