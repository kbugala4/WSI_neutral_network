from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from supporting_methods import ReLu

if __name__ == "__main__":

    dataset = load_digits()

    digits_data = dataset["data"]

    digits_labels = dataset["target"]

    print(digits_data / digits_data.max())

    print(digits_labels)

    print(digits_labels.shape)

    # encoder = OneHotEncoder()
    # encoder.fit(digits_labels)
    # new_arr = encoder.transform(digits_labels).toarray()
    # print(new_arr)

    new_labels = pd.get_dummies(digits_labels)
    print(new_labels)

    print(new_labels.shape)

    print(ReLu(digits_data - 1))
