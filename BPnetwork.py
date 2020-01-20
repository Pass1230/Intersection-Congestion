import numpy as np
import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_train = pd.read_csv('train.csv')


# calculate angle
def getAngle(df_train):
    angleValue = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
    a = df_train['EntryHeading']
    b = df_train['ExitHeading']
    angle = abs(angleValue[b] - angleValue[a])
    if angle > 180:
        angle -= 180
    return angle


df_train['angle'] = df_train.apply(getAngle, axis=1)

# create dummy vector
data = pd.get_dummies(df_train, columns=['Hour', 'Weekend', 'Month', 'City', 'angle'])
data = data.drop(
    ['RowId', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 'ExitStreetName', 'EntryHeading',
     'ExitHeading', 'Path'], axis=1)

y = np.array(data.loc[:8000, 'TotalTimeStopped_p20': 'DistanceToFirstStop_p80']).reshape(-1, 15)

x = np.array(data.loc[:8000, 'Hour_0': 'angle_180'])

# http://pybrain.org/docs/index.html.
# Divide the original dataset to training dataset and test dataset.
num = int(len(x) * 0.7)

# Normalize the data to the range from -1 to 1.
scalerx = MinMaxScaler()
xTrain = x[:num]
xTrain = scalerx.fit_transform(xTrain)

scalery = MinMaxScaler()
yTrain = y[:num]
yTrain = scalery.fit_transform(yTrain)

xTest = x[num:]
xTest = scalerx.transform(xTest)
yTest = y[num:]
yTest = scalery.transform(yTest)


# Initialize the original dataset and establish the neural network.
DS = SupervisedDataSet(x.shape[1], 15)

fnn = buildNetwork(DS.indim, 4, DS.outdim, bias=True, recurrent=True, hiddenclass=SigmoidLayer)

# Add the training data to DS.
for i in range(len(xTrain)):
    DS.addSample(xTrain[i], yTrain[i])

# Using BP model to train the data until convergence, the max train times is 30.
model = BackpropTrainer(fnn, DS, learningrate=0.01, verbose=True)
model.trainUntilConvergence(maxEpochs=30)

# Predict the validation dataset based on the established model.
values = []

for x in xTest:
    results = fnn.activate(x)
    final = scalery.inverse_transform(results.reshape(-1,15))
    values.append(final[0])

# Compute the root mean squared error(RMSE) to evaluate the effect of the model.
RMSE = sum(map(lambda x: x ** 0.5, map(lambda x, y: pow(x - y, 2), y[num:], values))) / float(len(xTest))
print("RMSE of predict :", RMSE)

