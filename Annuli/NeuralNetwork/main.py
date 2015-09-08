from ConvolutionalNN import evaluate_lenet5


if __name__ == '__main__':
    # evaluate_lenet5()
    # evaluate_lenet5(0.01, momentum=0.7, dataset='../DataGeneration/dataset-2chl.pkl')
    # evaluate_lenet5(0.01, momentum=0.7, dataset='../DataGeneration/dataset-2chl-20px.pkl')
    # evaluate_lenet5(0.01, momentum=0.7, dataset='../DataGeneration/dataset-2ch-60px.pkl')
    #To use confusion error and printing the confusion matrix use confusion_error=True, for Default, use default_error=True
    #This will produce the train, validate, test error curves,give the confusion matrix, save the weights to the same directory
    
    # evaluate_lenet5(0.1, momentum=0.7, dataset='/home/anupamajha/TUMSecondSemester/2013-DL-MrBlonde/Annuli/DataGeneration/datasetnobalance60px2ch.pkl', confusion_error=True, default_error=False, verbose=False, use_rmsprop=False)
    # evaluate_lenet5(0.02, momentum=0.7, dataset='../DataGeneration/dataset-2ch-multiclass-60px-unbal_test.pkl', confusion_error=True, default_error=True, verbose=True, use_rmsprop=True, n_epochs=20)
    evaluate_lenet5(0.02, momentum=0.7, dataset='../DataGeneration/dataset-4ch-60px-unbal_test.pkl', confusion_error=True, default_error=True, verbose=True, use_rmsprop=True, n_epochs=20)

    # evaluate_lenet5(0.1, n_epochs=3,
                    # dataset='/home/anupamajha/TUMSecondSemester/DeepLearning/2013-DL-MrBlonde/Annuli/DataGeneration/BalancedWithTestNotBalanced/tomtec2chamber.pkl',
                    # confusion_error=True, verbose=False, use_rmsprop=False)
