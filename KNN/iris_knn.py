import pandas as pd
from knn import KNN
from ml_statistics import calculate_accuracy
from ml_statistics import confusion_matrix

def prepare_data_set(data_path):

    df = pd.read_csv(data_path, header=None)

    df.columns = [
        "flower_id",
        "sepal_length",
        "sepal_width",
        "petal_length", 
        "petal_width", 
        "class",
    ]
    df = df.drop("flower_id", axis=1)
    return df



def main() -> None:
    k = 5
    knn = KNN(k = k)
    
    # trenownaie
    train_set_path = ("/home/mateusz/ML/KNN/iris.train")
    train_set = prepare_data_set(train_set_path)

    X = train_set.drop("class", axis=1) 
    X = X.values #macierz zmiennych niezależnych
    y = train_set["class"] 
    y = y.values #wektor zmiennych zależnych

    knn.train(X, y)

    # klasyfikacja
    validate_set_path = ("/home/mateusz/ML/KNN/iris.validate")
    validate_set =  prepare_data_set(validate_set_path)

    y_validate = validate_set["class"]
    predicted_classes = []

    validate_set = validate_set.drop("class", axis=1)
    X_validate = validate_set.values
    
    for sample in X_validate:
        predicted_classes.append(knn.predict(sample))

    M = confusion_matrix(target=y_validate, predicted=predicted_classes, n=3, 
                         index_dict= {"Iris-setosa": 0,"Iris-versicolor": 1,"Iris-virginica": 2})
    acc = calculate_accuracy(target=y_validate, predicted=predicted_classes)

    print("Confusion matrix")
    print(M)
    print(f"Accuracy: {acc}")

if __name__ == '__main__':
    main()
