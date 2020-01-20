import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('train.csv')
df_train.loc[:, 'DistanceToFirstStop_p20':'DistanceToFirstStop_p80'] = np.around(
    df_train.loc[:, 'DistanceToFirstStop_p20':'DistanceToFirstStop_p80'])
df_test = pd.read_csv('test.csv')


# calculate angle.
def getAngle(df_train):
    angleValue = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
    a = df_train['EntryHeading']
    b = df_train['ExitHeading']
    angle = abs(angleValue[b] - angleValue[a])
    if angle > 180:
        angle -= 180
    return angle


df_train['angle'] = df_train.apply(getAngle, axis=1)

data = pd.get_dummies(df_train, columns=['Hour', 'Weekend', 'Month', 'City', 'angle'])
data = data.drop(
    ['RowId', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 'ExitStreetName', 'EntryHeading',
     'ExitHeading', 'Path'], axis=1)

y = np.array(data.loc[:, 'DistanceToFirstStop_p50'])
x = np.array(data.loc[:, 'Hour_0': 'angle_180'])

df_test['angle'] = df_test.apply(getAngle, axis=1)
data1 = pd.get_dummies(df_test, columns=['Hour', 'Weekend', 'Month', 'City', 'angle'])
data1 = data1.drop(
    ['RowId', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 'ExitStreetName', 'EntryHeading',
     'ExitHeading', 'Path'], axis=1)

x1 = np.array(data1.loc[:, 'Hour_0': 'angle_180'])

# https://github.com/apachecn/lightgbm-doc-zh.
# Divide training set and validation set.
xTrainAll, xPredict, yTrainAll, yPredict = train_test_split(x, y, test_size=0.10, random_state=100)
xTrain, xTest, yTrain, yTest = train_test_split(xTrainAll, yTrainAll, test_size=0.2, random_state=100)
train_data = lgb.Dataset(data=xTrain, label=yTrain)
test_data = lgb.Dataset(data=xTest, label=yTest)

# Set parameters of the model.
param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'regression'}
param['metric'] = 'RMSE'

# Train the model based on the training dataset.
# Set early stopping rounds for the model.
num_round = 100
model = lgb.train(param, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=100)

# Save the model.
model.save_model('model.txt', num_iteration=model.best_iteration)

# Conduct cross-validation for the model.
lgb.cv(param, train_data, num_round, nfold=5, early_stopping_rounds=100)

# Predict the validation dataset based on the established model.
predictions = model.predict(xPredict, num_iteration=model.best_iteration)

# Compute the root mean squared error(RMSE) to evaluate the effect of the model.
RMSE = np.sqrt(mean_squared_error(yPredict, predictions))
print("RMSE of predict :", RMSE)

# A saved model can be loaded and predict the test data file.
bst = lgb.Booster(model_file='model.txt')  # init model
pred = bst.predict(x1, num_iteration=bst.best_iteration)








