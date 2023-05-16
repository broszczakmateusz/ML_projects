import pandas as pd
from perceptron import Perceptron 
from ml_statistics import calculate_accuracy
from ml_statistics import confusion_matrix

def prepare_data_set(data_path):

    df = pd.read_csv(data_path, header=None)

    df.columns = [
        "index",
        "variance",
        "skewness",
        "curtosis", 
        "entropy", 
        "class",
    ]
    df = df.drop("index", axis=1)

    # zamiana wartości klasy z 0 na -1
    df.loc[df["class"] == 0, ["class"]] = -1 

    return df


def main():
    # trenowanie
    train_data_path = ("/home/mateusz/ML/perceptron/data.train")
    train_data =  prepare_data_set(train_data_path)

    y_train = train_data["class"]

    train_data = train_data.drop("class", axis=1)
    X_train = train_data.values

    perceptron = Perceptron()
    perceptron.train(X = X_train, y = y_train)

    # klasyfikacja
    validate_data_path = ("/home/mateusz/ML/perceptron/data.validate")
    validate_data =  prepare_data_set(validate_data_path)

    y_validate = validate_data["class"]

    validate_data = validate_data.drop("class", axis=1)
    X_validate = validate_data.values

    predicted_classes = []
    for sample in X_validate:
        predicted_classes.append(perceptron.predict(sample))
    
    M = confusion_matrix(target=y_validate, predicted=predicted_classes, n=2, index_dict= {-1: 0, 1: 1})
    acc = calculate_accuracy(target=y_validate, predicted=predicted_classes)

    print("Confusion matrix")
    print(M)
    print(f"Accuracy: {acc}")


if __name__ == '__main__':
    main()
