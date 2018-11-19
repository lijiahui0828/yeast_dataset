from basicfunction import ANN, X_y_all
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    X_all, y_all_ = X_y_all()

    np.random.seed(1)
    model_all = ANN(output_dim=10)

    hist_all = model_all.fit(X_all, y_all_, batch_size=1, epochs=200, verbose=1)
    print("1-accuracy: {}".format(round(1-hist_all.history['acc'][-1],4)))
    print("CEloss: {}".format(round(hist_all.history['loss'][-1],4)))

    weight_all_final = model_all.layers[2].get_weights()

    print("Wi1 = {}".format(weight_all_final[0][0]))
    print("Wi2 = {}".format(weight_all_final[0][1]))
    print("Wi3 = {}".format(weight_all_final[0][2]))
    print("b = {}".format(weight_all_final[1]))