import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from  KNN.iris_data_preparation import split_iris_data
import KNN.knn as knn_algorithm
from perceptron.ml_statistics import calculate_accuracy
from perceptron.ml_statistics import confusion_matrix

def prepare_data_set(data_path):

    df = pd.read_csv(data_path, header=None)

    df.columns = [
        "index",
        "sepal_length",
        "sepal_width",
        "petal_length", 
        "petal_width", 
        "class",
    ]
    df = df.drop("index", axis=1)

    # zamiana słownej reprezentacji klasy liczbową
    df.loc[df["class"] == "Iris-setosa", ["class"]] = 0
    df.loc[df["class"] == "Iris-versicolor", ["class"]] = 1
    df.loc[df["class"] == "Iris-virginica", ["class"]] = 2

    return df


def main():
    n = 100
    k_params = [*range(1,21,1)]
    acc_for_k = np.zeros(len(k_params))

    for i in range(0,n):
        split_iris_data()
        train_data =  prepare_data_set("/home/mateusz/ML/KNN/iris.train")
        y_train = train_data["class"].values
        X_train = train_data.drop("class", axis=1).values

        validate_data =  prepare_data_set("/home/mateusz/ML/KNN/iris.validate")
        y_validate = validate_data["class"]
        X_validate = validate_data.drop("class", axis=1).values


        print(f'data_{i}')
        for k in k_params:
        # trenowanie KNN
            knn = knn_algorithm.KNN(k = k)
            knn.train(X_train, y_train)


        # klasyfikacja KNN
            predicted_classes = []
            for sample in X_validate:
                predicted_classes.append(knn.predict(sample))


            acc = calculate_accuracy(target=y_validate, predicted=predicted_classes)
            acc_for_k[k-1] += acc
            # print(f'k={k}\tacc={acc}')


    mean_acc_for_k = [x/n for x in acc_for_k]
    plt.plot(k_params, mean_acc_for_k, 'o')
    plt.xticks(k_params)
    plt.show()


if __name__ == '__main__':
    main()