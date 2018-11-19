from basicfunction import X_y_all,ANN
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def initial_weight():
    """
    initial the weight: the weight need calculating is 1, others are 0
    :return: the initialized weight
    """
    np.random.seed(3)
    W1 = np.zeros((8, 3))
    b1 = np.zeros((3,))
    W2 = np.zeros((3, 3))
    W2[0][0] = 1
    W2[1][0] = 1
    W2[2][0] = 1
    b2 = np.zeros((3,))
    b2[0] = 1
    W3 = np.zeros((3, 10))
    W3[0][0] = 1
    W3[1][0] = 1
    W3[2][0] = 1
    b3 = np.zeros((10,))
    b3[0] = 1
    weight_init = [W1, b1, W2, b2, W3, b3]
    return weight_init

if __name__ == "__main__":

    X_all, y_all_ = X_y_all()
    X_one = X_all.iloc[0:1, :]
    y_one = y_all_.iloc[0:1, :]
    print(X_one)
    print(y_one)

    weight_init = initial_weight()

    model_one = ANN(output_dim=10)
    model_one.set_weights(weight_init)
    model_one.fit(X_one, y_one, batch_size=1, epochs=1, verbose=1)

    weight_update= model_one.get_weights()
    print("Bias from output to second hidden (b_1^(3)): {}".format(weight_update[5][0]))
    print("Weights from output to second hidden (W_1i^(3)): {}".format([weight_update[4][0][0], weight_update[4][1][0], weight_update[4][2][0]]))
    print("Bias from second hidden to first hidden (b_1^(2)): {}".format(weight_update[3][0]))
    print("Weights from second hidden to first hidden (W_1i^(2)): {}".format([weight_update[2][0][0], weight_update[2][1][0], weight_update[2][2][0]]))