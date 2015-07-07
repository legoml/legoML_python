# Generic imports
from __future__ import division, print_function
from matplotlib import use
use('TkAgg')

from matplotlib.pyplot import draw, ioff, ion, pause, show, subplots

from numpy import atleast_2d, nan
from pprint import pprint
from os import chdir, path
from random import sample
from sys import argv, stdout
from time import sleep, time

# PyBrain imports
from pybrain import FeedForwardNetwork, LinearLayer, TanhLayer, SoftmaxLayer, FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer, RPropMinusTrainer
from pybrain.utilities import percentError

#from UtilityCode.mlPyUtil.CostFunctions import multi_class_cross_entropy


data = ClassificationDataSet(2, nb_classes=2, class_labels='AND')
# Assign Input data
train_data.setField('input', train_X)
# Assign Target data - *** note that it has to be 2-D column vector ***
train_data.setField('target', atleast_2d(train_y).T)
# Expand y integer class labels to sparse 0 / 1 matrix
train_data._convertToOneOfMany()

# Set up TEST_DATA - similar to above process for TRAIN_DATA
test_data = ClassificationDataSet(num_X_features, nb_classes=6, class_labels=y_class_labels)
test_data.setField('input', test_X)
test_data.setField('target', atleast_2d(test_y).T)
test_data._convertToOneOfMany()


# Create Feed-Forward Neural Network (FFNN)

input_layer = LinearLayer(num_X_features, name='InputLayer')
hidden_layer = TanhLayer(100, name='HiddenLayer')
output_layer = SoftmaxLayer(num_y_class_labels, name='OutputLayer')

ffnn = FeedForwardNetwork(name='MyFFNN')
ffnn.addInputModule(input_layer)
ffnn.addModule(hidden_layer)
ffnn.addOutputModule(output_layer)

input_to_hidden_connection = FullConnection(input_layer, hidden_layer, name='InputToHiddenConnection')
hidden_to_output_connection = FullConnection(hidden_layer, output_layer, name='HiddenToOutputConnection')

ffnn.addConnection(input_to_hidden_connection)
ffnn.addConnection(hidden_to_output_connection)

ffnn.sortModules()

# Print a description of FFNN
print('FEED-FORWARD NEURAL NETWORK (FFNN) ARCHITECTURE:')
print(ffnn)
stdout.flush()

print(vars(ffnn))

print(len(ffnn.params))

# Set training parameters

# *******************************************************
# *** SET YOUR TRAINING SETTINGS HERE                 ***
trainer = BackpropTrainer(ffnn,
                          dataset=train_data,
                          learningrate=.1,
                          lrdecay=.99,  # the bigger, the slower the learning rate decay
                          momentum=0.,
                          weightdecay=0.0,   # the bigger, the faster the weight decay
                          batchlearning=False,
                          verbose=True)

num_train_epochs = 10
# *******************************************************


# Training FNN

ion()
train_epochs_list = range(num_train_epochs)
train_errors_list = num_train_epochs * [nan]
test_errors_list = num_train_epochs * [nan]
fig, axes = subplots()
train_errors_plot, = axes.plot(train_epochs_list, train_errors_list)
test_errors_plot, = axes.plot(train_epochs_list, test_errors_list)

print('Training FFNN...')
start_train_time = time()



for train_epoch in range(num_train_epochs):
    trainer.trainEpochs(1)

    train_predicted_output = ffnn.activateOnDataset(train_data)
    train_error = multi_class_cross_entropy(train_predicted_output, train_data['target'])
    train_percent_error = percentError(trainer.testOnClassData(), train_data['class'])

    test_predicted_output = ffnn.activateOnDataset(test_data)
    test_error = multi_class_cross_entropy(test_predicted_output, test_data['target'])
    test_percent_error = percentError(trainer.testOnClassData(dataset=test_data), test_data['class'])

    print("epoch: %4d" % trainer.totalepochs,
          "  train error: %5.2f%%" % train_percent_error,
          "  test error: %5.2f%%" % test_percent_error)

    train_errors_list[train_epoch] = train_error
    test_errors_list[train_epoch] = test_error
    train_errors_plot.set_ydata(train_errors_list)
    test_errors_plot.set_ydata(test_errors_list)
    axes.relim()
    axes.autoscale_view(True, True, True)
    fig.canvas.draw()
    draw()
    pause(1e-1)
    #sleep(1e-1)

end_train_time = time()
print('   (training took', '{:,}'.format(round(end_train_time - start_train_time, 0)), 'seconds)')

ioff()
show()

