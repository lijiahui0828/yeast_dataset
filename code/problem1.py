from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from basicfunction import read_data

def Remove_Outliers_IF(dat):
    """
    Detect Outliers by Isolation Forest and Remove them
    :param dat: data that need to remove outliers
    :return isf_outliers_ind: the index of outliers
    :return data: the data after removing outliers
    """
    X = dat.iloc[:, 1:9]
    n_samples = dat.shape[0]
    isf = IsolationForest(max_samples=n_samples, contamination='auto', behaviour='new', random_state=1)
    isf.fit(X)
    isf_outliers_ind = isf.predict(X) == -1
    data = dat[~isf_outliers_ind]
    return isf_outliers_ind, data

def Remove_Outliers_LOF(dat, outlier_fraction):
    """
    Detect Outliers by LOF and Remove them
    :param dat: data that need to remove outliers
    :param outlier_fraction: the contamination parameter in the LOF method
    :return lof_outliers_ind: the index of outliers
    :return data: the data after removing outliers
    """
    X = dat.iloc[:, 1:9]
    lof = LocalOutlierFactor(contamination=outlier_fraction)
    lof_outliers_ind = lof.fit_predict(X) == -1
    data = dat[~lof_outliers_ind]
    return lof_outliers_ind, data


if __name__ == "__main__":
    """
    get the result of problem1 and print
    """
    dat = read_data()
    isf_outliers_ind, data = Remove_Outliers_IF(dat)
    print("There are {} outliers by Isolation Forest method.".format(sum(isf_outliers_ind)))

    # LOF
    lof_outliers_ind = Remove_Outliers_LOF(dat, 'auto')[0]
    print("There are {} outliers by LOF method.".format(sum(lof_outliers_ind)))

    # compare if they agree
    outlier_fraction = round(sum(isf_outliers_ind) / len(dat), 3)
    print("Set the same comtamination : {} to compare.".format(outlier_fraction))
    lof_outliers_ind = Remove_Outliers_LOF(dat, outlier_fraction)[0]

    same_count = sum([isf_outliers_ind[i] and lof_outliers_ind[i] for i in range(len(dat))])
    print("Among 58 outliers detected by IF, there are {} are detected by LOF too, about {}%.\
    ".format(same_count, round(same_count / 58, 2) * 100))

    print("The new, revised dataset has {} samples, and only has {} different classes since all the 5 samples of "
          "'ERL' class have been dropped.".format(data.shape[0], len(data.loc_site.value_counts())))
