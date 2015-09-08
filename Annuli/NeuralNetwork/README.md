
To run ConvolutionalNN.py, change the call to the training method in `main.py` according to your needs.

The current results obtained from running this program on 2 chamber and 4 chamber Annuli data are as follows:


4 chamber:

Best validation score of 2.414506 %,with test performance  2.428028 %


2 Chamber:

Best validation score of 2.741667 %,with test performance 2.808497 %



The neuralnetwork.py has two types of nn: 
1) stochastic gradient descent
2) rmsprop

To change the data set between "tomtec2chamber.pkl" and "tomtec4chamber.pkl", just uncomment the corresponding method call in main method with path to dataset.

The following changes have been made to ConvolutionalNN.py:

1) To run it with Q2 error metric pass option default_error = True in the call evaluate_lenet5(), use this if you just need error
and not bothered to find confusion matrix

2) To run with confusion error metric pass option confusion_error = True in the call evaluate_lenet5()

Right now, both are set to True, but I think now we don't need the old error so, you can set default_error = False.

3) It prints training error, validation and testing error.

4) It plots train, validate and test error plots
 
5) It saves the weights in the same directory.

