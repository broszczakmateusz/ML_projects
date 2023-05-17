import pandas as pd
import numpy as np
from perceptron import Perceptron 
from ml_statistics import calculate_accuracy
from ml_statistics import confusion_matrix

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

def make_binary_classes(classes, selected_class):
    # dla klasy szukanej przypisuje 1, dla pozostałych -1
    for i in range(len(classes)):
        if  classes[i] == selected_class:
            classes[i] = 1
        else:
            classes[i] = -1

    return classes


def main():
    flower_classes = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
    }

    train_data_path = ("/home/mateusz/ML/KNN/iris.train")
    validate_data_path = ("/home/mateusz/ML/KNN/iris.validate")

    perceptrons = []
    # trenowanie
    for i in flower_classes:
        train_data =  prepare_data_set(train_data_path)
        y_train = train_data["class"]
        y_train = make_binary_classes(np.array(y_train), i)

        train_data = train_data.drop("class", axis=1)
        X_train = train_data.values

        perceptron = Perceptron()
        perceptron.train(X = X_train, y = y_train)
        perceptrons.append(perceptron)


    # klasyfikacja
    validate_data =  prepare_data_set(validate_data_path)

    y_validate = validate_data["class"]

    validate_data = validate_data.drop("class", axis=1)
    X_validate = validate_data.values    

    predicted_classes = []
    for sample in X_validate:
        predicted_for_class = np.zeros(3)
        for i in range(0,len(perceptrons)):
            predicted_for_class[i] = perceptrons[i].one_vs_all(sample) 
        
        predicted = predicted_for_class.argsort()[-1]
        predicted_classes.append(predicted)
    
    M = confusion_matrix(target=y_validate, predicted=predicted_classes, n=3, 
                         index_dict= {0: 0, 1: 1, 2: 2})
    acc = calculate_accuracy(target=y_validate, predicted=predicted_classes)

    print("Confusion matrix")
    print(M)
    print(f"Accuracy: {acc}")

if __name__ == '__main__':
    main()

