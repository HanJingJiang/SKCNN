

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import csv
from keras.datasets import mnist
from numpy import *
import numpy as np
np.random.seed(1337)
from keras.models import Model
from tqdm import tqdm
from sklearn.svm import SVC


data = []
data1 = ones((1, 2532), dtype=int)
# 2532 1933
data2 = zeros((1, 2532))
data.extend(data1[0])
data.extend(data2[0])
SampleLabel = data
print(len(SampleLabel))

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return
SampleFeature = []
ReadMyCsv(SampleFeature, 'dense1.csv')

x_train, x_test, y_train, y_test = train_test_split(SampleFeature, SampleLabel, test_size=0.2)
print('Start training the model.')

cv = StratifiedKFold(n_splits=10)
SampleFeature = np.array(SampleFeature)
SampleLabel = np.array(SampleLabel)
permutation = np.random.permutation(SampleLabel.shape[0])
SampleFeature = SampleFeature[permutation, :]
SampleLabel = SampleLabel[permutation]


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
num=0


for train, test in cv.split(SampleFeature, SampleLabel):
    model = RandomForestClassifier(n_estimators=100,max_depth=100)
    predicted = model.fit(SampleFeature[train], SampleLabel[train]).predict_proba(SampleFeature[test])


    fpr, tpr, thresholds = roc_curve(SampleLabel[test], predicted[:, 1])

    predicted1 = model.predict(SampleFeature[test])
    num = num + 1
    print("==================", num, "fold", "==================")
    print('Test accuracy: ', accuracy_score(SampleLabel[test], predicted1))
    print(classification_report(SampleLabel[test], predicted1, digits=4))
    print(confusion_matrix(SampleLabel[test], predicted1))

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))

    i += 1
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=1.5, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


