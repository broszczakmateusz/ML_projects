import pandas as pd
import numpy as np

df = pd.read_csv("/home/mateusz/ML/KNN/iris.data", header=None)


train, validate, test = np.split(df.sample(frac = 1), [int(.6*len(df)), int(.8*len(df))])

train = pd.DataFrame(train).to_csv(header=None)
validate = pd.DataFrame(validate).to_csv(header=None)
test = pd.DataFrame(test).to_csv(header=None)

data_sets = [train, validate, test]
output_files = ["/home/mateusz/ML/KNN/iris.train", "/home/mateusz/ML/KNN/iris.validate", "/home/mateusz/ML/KNN/iris.test"]

for i in range(0,3):
    with open(output_files[i], 'w') as f_out:
        f_out.writelines(data_sets[i])