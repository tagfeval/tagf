from numpy import mean
from numpy import std
from numpy import argmax, stack, array, sqrt
from pandas import read_csv, DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from keras import optimizers
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from sklearn.model_selection import train_test_split
from pandas import concat
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
	if not title:
		if normalize:
			title = 'Normalized Confusion Matrix'
		else:
			title = 'Confusion Matrix without normalization'
	#compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	#only compute on labels found within the data
	classes = list(unique_labels(y_true, y_pred))
	if normalize:
		cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
		print('Normalized Confusion Matrix')
	else:
		print('Confusion Matrix without normalization')
	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	#show all ticks
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True Labels', xlabel='Predicted Labels')
	#rotate ticks and set alignment
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	#loop over data dimensions and create text annotations
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax







def feature_importances(X, reg, feats=5):
    feature_imps = [(imp, X.columns[i]) 
                    for i, imp in enumerate(reg.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
    print('-----\n')
    return [f for f in feature_imps[:feats]]



#load data
data = read_csv('..//spss_new.csv', header=0, index_col=0)
df = data[['Analysis', 'Urban?', 'Univariate?.1', 'DCM_Bin', 'DS_Large?', 'PH', 'Real_Time_True?', 'DAM_1']]



X = df.iloc[:,0:7]
#X = X[['Analysis', 'DCM_Bin', 'DS_Large?', 'PH', 'Real_Time_True?']]
y = df.iloc[:,-1]

X = array(X)
y = array(y)

print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)


col_list = list(df)
col_list.pop()

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=27)
trainX, trainy = sm.fit_resample(trainX, trainy)
print('Resampled dataset shape %s' % Counter(trainy))
tX = DataFrame(trainX)
tX_feat = DataFrame(trainX, columns=col_list)

   
# run the experiment
#run_experiment(1)

#create decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(trainX,trainy)

#Predict the response for test dataset
y_pred = clf.predict(testX)

#feat_cols = ['Analysis', 'DCM', 'Size_Large', 'PH', 'Real']
print("Accuracy:",metrics.accuracy_score(testy, y_pred))
dot_data = StringIO()
feature_cols = ['Analysis Level', 'Urban?', 'Univariate?', 'DCM','Dataset Size','Pred. Horizon','Realtime']
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=feature_cols, rounded=True, special_characters=True, class_names=['0','1', '2','3','4','5','6'])


reg = RandomForestClassifier(n_estimators=10)
reg.fit(trainX, trainy)
importances = reg.feature_importances_
indices = np.argsort(importances)
preds = reg.predict(testX)
scores = (mean_squared_error(preds, testy))
feat = feature_importances(tX_feat, reg, feats=20)


cm = confusion_matrix(testy, preds)

y_pred = preds
y_test = testy

class_names = list(unique_labels(y_test, y_pred))
print(class_names)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False, title='Normalized Confusion matrix for Test Dataset')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred, target_names=['0','1','2','3','4','5','6']))

#plot feature importance

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [col_list[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()