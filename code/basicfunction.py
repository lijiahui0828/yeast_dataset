import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def read_data():
    """
    read the data.
    :return dat: the data in DataFrame type.
    """
    colnames = ['seq_name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'loc_site']
    dat = pd.read_table('yeast.data', delim_whitespace=True, header=None,
                        names=colnames)
    return dat


def ANN(output_dim, num_hidden_layer=2, num_nodes_hidden=3):
    """
    construct the ANN.
    :param output_dim: the dim of output
    :param num_hidden_layer: the number of hidden layers
    :param num_nodes_hidden: the number of nodes in each hidden layer
    :return: the constructed ANN model
    """
    model = Sequential()
    if num_hidden_layer == 1:
        model.add(Dense(units=output_dim, input_dim=8, activation='softmax'))
    else:
        model.add(Dense(units=num_nodes_hidden, input_dim=8, activation='sigmoid'))
        model.add(Dense(units=num_nodes_hidden, activation='sigmoid'))
        if num_hidden_layer >= 3:
            model.add(Dense(units=num_nodes_hidden, activation='sigmoid'))

        model.add(Dense(units=output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def To_Categorical(y_origin, class_num):
    """
    To converts a class vector into a DataFrame with as many columns as there are classes, One-hot-Encoder.
    :param y_origin: the class vector need to convert.
    :param class_num: the number of classes.
    :return: the dataframe we convert to.
    """
    y_encoded, y_categories = y_origin.factorize()
    y_cat = to_categorical(y_encoded, class_num)
    y_cat = pd.DataFrame(y_cat, columns= y_categories)
    return(y_cat)

def X_y_all():
    """
    read all the data, split it into X and y, convert y to categorical
    :return X_all: X of all 1484 samples
    :return y_all_: y of all 1484 samples after converting to categorical
    """
    dat = read_data()
    X_all = dat.iloc[:, 1:9]
    y_all = dat.loc_site
    y_all_ = To_Categorical(y_all, 10)
    return X_all, y_all_