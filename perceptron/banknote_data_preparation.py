import pandas as pd
import numpy as np

df = pd.read_csv("/home/mateusz/ML/perceptron/data_banknote_authentication.txt", header=None)

seed = 15
train, validate, test = np.split(df.sample(frac = 1, random_state = seed), [int(.6*len(df)), int(.8*len(df))])

train = pd.DataFrame(train).to_csv(header=None)
validate = pd.DataFrame(validate).to_csv(header=None)
test = pd.DataFrame(test).to_csv(header=None)

data_sets = [train, validate, test]
output_files = ["/home/mateusz/ML/perceptron/data.train", "/home/mateusz/ML/perceptron/data.validate", "/home/mateusz/ML/perceptron/data.test"]

for i in range(0,3):
    with open(output_files[i], 'w') as f_out:
        f_out.writelines(data_sets[i])