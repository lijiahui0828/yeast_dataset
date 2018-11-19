import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from basicfunction import read_data, To_Categorical, ANN
from problem1 import Remove_Outliers_IF


if __name__ == "__main__":
    dat = read_data()
    data = Remove_Outliers_IF(dat)[1]
    X = data.iloc[:, 1:9]
    y = data.loc_site
    y_ = To_Categorical(y, 9)
    X_new = np.array([0.52, 0.47,0.52,0.23,0.55,0.03,0.52,0.39]).reshape(1,-1)
    np.random.seed(2)
    model_new = ANN(output_dim=9, num_hidden_layer=2, num_nodes_hidden= 9 )
    model_new.fit(X, y_, batch_size=1, epochs=200, verbose=1)
    prob = model_new.predict(X_new)
    print("It belongs to {} class, with probability = {}.".format(y_.columns[np.argmax(prob)], prob[0][np.argmax(prob)]))
    df_prob = pd.DataFrame({'class':y_.columns, 'pred_prob':prob[0]})
    print("The probability of every class is {}.".format(df_prob))