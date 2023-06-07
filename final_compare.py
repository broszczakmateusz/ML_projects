import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics


import perceptron.perceptron as percep
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

def make_binary_classes(classes, selected_class):
    # dla klasy szukanej przypisuje 1, dla pozostałych -1
    for i in range(len(classes)):
        if  classes[i] == selected_class:
            classes[i] = 1
        else:
            classes[i] = -1

    return classes

def algorithm_stats(algorithm_name, y_validate, predicted_classes):
    labels = {0: "Iris-setosa",1: "Iris-versicolor",2: "Iris-virginica"}

    M = confusion_matrix(target=y_validate, predicted=predicted_classes, n=3, index_dict= {0: 0, 1: 1, 2: 2}, labels=labels)
    acc = calculate_accuracy(target=y_validate, predicted=predicted_classes)
    print(f"\t***{algorithm_name}***")
    print("Confusion matrix")
    print(M)
    print(f"Accuracy: {acc}\n")

def main():
    train_data =  prepare_data_set("/home/mateusz/ML/KNN/iris.train")
    y_train = train_data["class"].values
    X_train = train_data.drop("class", axis=1).values

    validate_data =  prepare_data_set("/home/mateusz/ML/KNN/iris.validate")
    y_validate = validate_data["class"]
    X_validate = validate_data.drop("class", axis=1).values

    # trenowanie KNN
    k = 5
    knn = knn_algorithm.KNN(k = k)
    knn.train(X_train, y_train)


    # trenowanie ONE-VS-REST PERCEPTRON
    flower_classes = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    perceptrons = []
    for i in flower_classes:
        binary_y_train = make_binary_classes(np.array(y_train), i)

        perceptron = percep.Perceptron()
        perceptron.train(X = X_train, y = binary_y_train)
        perceptrons.append(perceptron)

    # trenowanie MLP
    MLP = MLPClassifier(random_state=1, max_iter=300).fit(X_train.tolist(), y_train.tolist())

    # trenowanie drzewo decyzyjne
    decision_tree = DecisionTreeClassifier().fit(X_train.tolist(), y_train.tolist())

    # trenowanie SVM
    SVM = SVC().fit(X_train.tolist(), y_train.tolist())

    # klasyfikacja KNN
    predicted_classes = []
    for sample in X_validate:
        predicted_classes.append(knn.predict(sample))

    algorithm_stats("KNN", y_validate, predicted_classes)

    # klasyfikacja ONE-VS-REST PERCEPTRON
    predicted_classes = []
    for sample in X_validate:
        predicted_for_class = np.zeros(3)
        for i in range(0,len(perceptrons)):
            predicted_for_class[i] = perceptrons[i].one_vs_all(sample) 
        
        predicted = predicted_for_class.argsort()[-1]
        predicted_classes.append(predicted)
    
    algorithm_stats("1-vs-rest PERCEPTRON", y_validate, predicted_classes)

    # klasyfikacja MLP
    predicted_classes = MLP.predict(X=X_validate)
    algorithm_stats("MLP", y_validate, predicted_classes)

    # klasyfikacja drzewo decyzyjne
    predicted_classes = decision_tree.predict(X=X_validate)
    algorithm_stats("Decision tree", y_validate, predicted_classes)

    # klasyfikacja SVM
    predicted_classes = SVM.predict(X=X_validate)
    algorithm_stats("SVM", y_validate, predicted_classes)

if __name__ == '__main__':
    main()

# todo
# opisać wpływ k na accuracy
# opisać metryki
# opisać 3 klasyfikatory

# wpływ hiperparametrów na accuracy
# opisać badania
# przedstawić wnioski