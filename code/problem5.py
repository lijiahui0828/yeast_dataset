import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from basicfunction import ANN
from problem2 import  Remove_Outliers_and_Split_Train_Test

if __name__ == "__main__":
    nums_hidden_layer = [1, 2, 3]
    nums_nodes_hidden = [3, 6, 9, 12]

    df = pd.DataFrame(np.zeros(12).reshape(3, 4))
    df.index = nums_hidden_layer
    df.columns = nums_nodes_hidden


    X_train, X_test, y_train_, y_test_ = Remove_Outliers_and_Split_Train_Test()

    np.random.seed(1)
    for i in range(3):
        for j in range(4):
            model_sweep = ANN(output_dim=9, num_hidden_layer=nums_hidden_layer[i],
                                    num_nodes_hidden=nums_nodes_hidden[j])
            hist = model_sweep.fit(X_train, y_train_, batch_size=1, epochs=200, verbose=1,
                                   validation_data=(X_test, y_test_))
            df.iloc[i, j] = hist.history['val_loss'][-1]

    print (df)