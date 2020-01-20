from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import metrics
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


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

data = pd.get_dummies(df_train, columns=['Hour', 'Weekend', 'Month', 'City', 'angle'])
data = data.drop(
    ['RowId', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 'ExitStreetName', 'EntryHeading',
     'ExitHeading', 'Path'], axis=1)

y = np.array(data.loc[: data.shape[0] - 2, 'DistanceToFirstStop_p80']).reshape(-1, 1)
x = np.array(data.loc[: data.shape[0] - 2, 'Hour_0': 'angle_180'])

num = int(len(x) * 0.5)

xTrain = x[:num]
yTrain = y[:num]
xTest = x[num:]
yTest = y[num:]

# Establish random forests model.
model = RandomForestRegressor()

# Train the model with training dataset.
model.fit(xTrain, yTrain)

# Predict the validation dataset based on the established model.
predictions = model.predict(xTest)

# Compute the root mean squared error(RMSE) to evaluate the effect of the model.
RMSE = metrics.mean_squared_log_error(predictions, yTrain)
print("RMSE of predict :", RMSE)

