# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
#
# import numpy as np
# import pandas as pd
#
# # ----- -----
# def main():
#     # sel = int (sys.argv[1])
#     msft = pd.read_csv('phoneAcceleromenterDistractionExp2Part1_05.csv')
#     pd.plotting.register_matplotlib_converters()
#
#     # with cbook.get_sample_data('phoneAcceleromenterDistractionExp2Part1_05.csv') as file:
#     #     msft = pd.read_csv(file, parse_dates=['Date'])
#     msft.plot(0, [1, 2], subplots=True)
#     print("Done")







import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import itertools
import tensorflow as tf


from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None

def main1():
    headers = ['Date','Unfiltered1','Filtered1','Unfiltered2','Filtered2']
    df = pd.read_csv('FilteredUnfiltered.CSV',names=headers)
    print (df)

    df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
    x = df['Date']
    y1 = df['Unfiltered1']
    y2 = df['Filtered1']
    y3 = df['Unfiltered2']
    y4 = df['Filtered2']

    # plot
    plt.plot(x,y1, color='red', label='Unfiltered1')
    plt.plot(x,y2, color='green', label='Filtered1')
    plt.plot(x,y3, color='orange', label='Unfiltered2')
    plt.plot(x,y4, color='blue', label='Filtered2')
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    # plt.legend(['Unfiltered'])
    plt.legend(loc="lower right")
    plt.show()


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    filenames += ['AccelerometerX_' + group + '.txt', 'AccelerometerY_' + group + '.txt', 'AccelerometerZ_' + group + '.txt']
    # filenames += ['OrientationX_' + group + '.txt', 'OrientationY_' + group + '.txt', 'OrientationZ_' + group + '.txt']
    filenames += ['RotationRateX_' + group + '.txt', 'RotationRateY_' + group + '.txt', 'RotationRateZ_' + group + '.txt']
    filenames += ['UserAccelerationX_' + group + '.txt', 'UserAccelerationY_' + group + '.txt', 'UserAccelerationZ_' + group + '.txt']
    # filenames += ['QuaternionX_' + group + '.txt', 'QuaternionY_' + group + '.txt', 'QuaternionZ_' + group + '.txt']
    # filenames += ['GravityX_' + group + '.txt', 'GravityY_' + group + '.txt', 'GravityZ_' + group + '.txt']
    # filenames += ['watchAccelerometerX_' + group + '.txt', 'watchAccelerometerY_' + group + '.txt', 'watchAccelerometerZ_' + group + '.txt']
    # filenames += ['watchOrientationX_' + group + '.txt', 'watchOrientationY_' + group + '.txt', 'watchOrientationZ_' + group + '.txt']
    # filenames += ['watchRotationRateX_' + group + '.txt', 'watchRotationRateY_' + group + '.txt', 'watchRotationRateZ_' + group + '.txt']
    # filenames += ['watchUserAccelerationX_' + group + '.txt', 'watchUserAccelerationY_' + group + '.txt', 'watchUserAccelerationZ_' + group + '.txt']
    # filenames += ['watchQuaternionX_' + group + '.txt', 'watchQuaternionY_' + group + '.txt', 'watchQuaternionZ_' + group + '.txt']
    # filenames += ['watchGravityX_' + group + '.txt', 'watchGravityY_' + group + '.txt', 'watchGravityZ_' + group + '.txt']

    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'DatasetFull/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'DatasetFull/')
    print(testX.shape, testy.shape)
    # zero-offset class valuesF
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 100
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    other, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    result = model.predict(testX, batch_size=batch_size, verbose=0)

    presiction = list(map(np.argmax, result))
    actual = list(map(np.argmax, testy))

    tp, tn, fp, fn, p, n = calculate_metrics(presiction, actual)
    # print('tp: %d\ttn: %d\tfp: %d\tfn: %d\tp: %d, n: %d' % (tp, tn, fp, fn, p, n))

    accuracy =  ((tp + tn) / (p + n))
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    f1 =  ((2*tp) / ((2*tp) + fp + fn))

    # print("undone")
    # print(presiction)
    # print(actual)

    # print("Result: ")
    # print(result)
    # print("Accuracy: ")
    # print(accuracy)

    # print("Begin try")
    # print(testy)

    # dot_img_file = 'model_1.png'
    # plot_model(model, to_file=dot_img_file, show_shapes=True)

    tf.keras.utils.plot_model(model, to_file='pleassse.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96)

    if (accuracy > 0.93) and (precision > 0.99) and (recall > 0.87) and (f1 > 0.93):
        model.save('saved_model/my_model')


    return accuracy, precision, recall, f1, tp, tn, fp, fn

def calculate_metrics(prediction, actual):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    p = 0
    n = 0
    for i in range(len(actual)):
        if prediction[i] == actual[i]:
            if prediction[i]  == 1:
                tn += 1
                n += 1
            else:
                tp +=1
                p += 1
        else:
            if prediction[i] == 0:
                fp += 1
                n += 1
            else:
                fn += 1
                p += 1

    return tp, tn, fp, fn, p, n


# summarize scores
def summarize_results(accuracies, precisions, recalls, f1s):
    m, s = mean(accuracies), std(accuracies)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m*100, s*100))
    print(accuracies)

    m, s = mean(precisions), std(precisions)
    print('Precision: %.3f%% (+/-%.3f)' % (m*100, s*100))
    print(precisions)

    m, s = mean(recalls), std(recalls)
    print('Recall: %.3f%% (+/-%.3f)' % (m*100, s*100))
    print(recalls)

    m, s = mean(f1s), std(f1s)
    print('F1: %.3f%% (+/-%.3f)' % (m*100, s*100))
    print(f1s)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    accuracies = list()
    precisions = list()
    recalls = list()
    f1s = list()
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for r in range(repeats):
        accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate_model(trainX, trainy, testX, testy)
        # score = score * 100.0
        print('>#%d:' % (r+1))
        print('Accuracy: %f Prcision: %f, Recall: %f, f1: %f' % (accuracy, precision, recall, f1))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
    # summarize results
    summarize_results(accuracies, precisions, recalls, f1s)
    print('tp: %d, tn: %d, fp: %d, fn: %d' % (total_tp, total_tn, total_fp, total_fn))
    plot_confusion_matrix(cm=np.array([[total_tp, total_fp],
                                       [total_fn, total_tn]]),
                          normalize=False,
                          target_names=['Distracted', 'Not-Distracted'],
                          title="Confusion Matrix")

def main2():
    run_experiment()


# ----- ----- ----- ----- ----- -----
if __name__ == '__main__':
    main2()

