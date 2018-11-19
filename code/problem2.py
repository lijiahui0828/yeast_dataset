import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.callbacks import LambdaCallback
from basicfunction import  read_data, ANN, To_Categorical
from problem1 import Remove_Outliers_IF

def Remove_Outliers_and_Split_Train_Test():
    """
    read the data, remove the outliers, and split it into train set and test set
    :return X_train, X_test, y_train_, y_test_: X_train, X_test, y_train_
    (after to categorical), y_test_ (after to categorical)
    """
    dat = read_data()
    data = Remove_Outliers_IF(dat)[1]
    X = data.iloc[:, 1:9]
    y = data.loc_site
    y_ = To_Categorical(y, 9)
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y_, test_size=0.3, random_state=1)
    return X_train, X_test, y_train_, y_test_

def weight_CYT(model_weights):
    """
    get the last layer weight of CYT
    :param model_weights: all the last layers weight
    :return weights_CYT: the last layers "CYT" nodes weight (4 weights)
    """
    weights_CYT = [model_weights[1][ind_CYT]]
    for i in range(3):
        weights_CYT.append(model_weights[0][i][ind_CYT] )
    return weights_CYT

def predict_CYT(model):
    """
    the output of the CYT node
    :param model: model to predict
    :return: the output of CYT node in train and test set
    """
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_train_CYT = [pred_train[i][ind_CYT] for i in range(len(X_train))]
    pred_test_CYT = [pred_test[i][ind_CYT] for i in range(len(X_test))]
    return pred_train_CYT, pred_test_CYT

if __name__ == "__main__":

    X_train, X_test, y_train_, y_test_ = Remove_Outliers_and_Split_Train_Test()

    np.random.seed(1)
    model = ANN(output_dim=9)

    ind_CYT = np.array(range(9))[y_train_.columns == 'CYT'][0]

    iter_weights = []
    iter_train_error = []
    iter_test_error = []

    save_weight = LambdaCallback(on_epoch_begin=lambda epoch, logs: iter_weights.append(
        weight_CYT(model.layers[2].get_weights())))
    save_train_error = LambdaCallback(on_epoch_begin=lambda epoch, logs: iter_train_error.append(
        log_loss(y_train_.CYT, predict_CYT(model)[0])))
    save_test_error = LambdaCallback(on_epoch_begin=lambda epoch, logs: iter_test_error.append(
        log_loss(y_test_.CYT, predict_CYT(model)[1])))

    model.fit(X_train, y_train_, batch_size=1, epochs=200, verbose=1, validation_data=(X_test, y_test_),
              callbacks=[save_weight, save_train_error, save_test_error])

    iter_weights.append(weight_CYT(model.layers[2].get_weights()))
    iter_train_error.append(log_loss(y_train_.CYT, predict_CYT(model)[0]))
    iter_test_error.append(log_loss(y_test_.CYT, predict_CYT(model)[1]))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iter_weights)
    plt.title("Weight Values Per Iteration for the Last Layer for CYT class", size=15)
    plt.xlabel("epoch")
    plt.legend(('bias', 'weight_31', 'weight_32', 'weight_33'))

    plt.subplot(1, 2, 2)
    plt.plot(iter_train_error, label='train error')
    plt.plot(iter_test_error, label='test error')
    plt.xlabel("epoch")
    plt.ylabel("CEloss")
    plt.title("Training and Test Error Per Iteration for CYT class", size=15)
    plt.legend()

    plt.show()




