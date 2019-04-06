import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import csv
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

import numpy as np
SampleFeature = []
ReadMyCsv(SampleFeature, "cSampleFeature.csv")
SampleFeature = np.array(SampleFeature)
print('SampleFeature',len(SampleFeature))
print('SampleFeature[0]',len(SampleFeature[0]))
x = SampleFeature #
data = []
data1 = np.ones((2532,1), dtype=int)
data2 = np.zeros((2532,1))
y=np.concatenate((data1,data2),axis=0)
print(y.shape)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train=x_train.reshape(-1,1,1072,1)
x_test=x_test.reshape(-1,1,1072,1)
x = x.reshape(-1,1,1072,1)
print(x_train.shape)
print(x_test.shape)
batch_size=32
epochs=2
model = Sequential()
return_sequences=True

model.add(Conv2D(32, (16, 16),strides=(2,2), activation='relu', padding='same', data_format='channels_last',name='layer1_con1',input_shape=(1,1072,1)))
model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer1_pool'))

model.add(Flatten())
model.add(Dense(64, activation='relu',  name='Dense-2'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()
import numpy as np
model.fit(x_train, y_train, epochs=20, batch_size=10,validation_split=0.1)
from keras.models import Model
dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense-2').output)
dense1_output = dense1_layer_model.predict(x)
print(dense1_output.shape)
print(dense1_output[0])
storFile(dense1_output, 'cdense1.csv')