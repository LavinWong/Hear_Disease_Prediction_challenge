import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import svm
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


#Load the data
data = pd.read_csv('heart.csv')

#Simple view of data.
print("The first five lines for dataset:")
print(data.head(5))

#Simple view of data shape, infomation and data types.
print("The shape of dataset:", data.shape)
print(data.info())
print(data.dtypes)
print(data.describe())
columnname = data.columns.values.tolist()
#create graph for distribution of the data
for i in range(13):
    sns.displot(data[columnname[i]])

#Check is any miss values in the data set.
print('The number of null value:',data.isnull().any().sum())

#Check is any duplicate values in the data set.
duplicate_row_data = data[data.duplicated()]
print("The duplicate number is", duplicate_row_data.shape)

#Check the correlation between each columns and other columns, and follow the correlation to sorted it.
newdata = data.corr()
for i in range(14):
    print(newdata.iloc[i].sort_values(ascending=False))

#Use heatmap to draw the graph to show the correlation between the first 6 columns.
sns.heatmap(newdata.iloc[0:6,0:6], annot=True, square=True)
plt.xlabel('column')
plt.ylabel('column')
plt.savefig('./Correlation between different features.jpg')
plt.show()
#Use heatmap to draw the graph to show the correlation between the 6 to 12 columns.
sns.heatmap(newdata.iloc[6:12,6:12], annot=True, square=True)
plt.xlabel('column')
plt.ylabel('column')
plt.savefig('./Correlation between different features.jpg')
plt.show()

#Use variance selection model to irrelevant or redundant features.
features_new = feature_selection.VarianceThreshold(threshold=2).fit_transform(data)
print('After the variance selection,the data set size:',features_new.shape)



#LR
x = data.iloc[:,:13].values
y = data.iloc[:,-1].values

#Check the data balance
count1 = 0
count2 = 0

for i in y:
    if i == 0:
        count1 +=1
    else:
        count2 +=1
print("The number of label 0:",count1, "The number of label 1:", count2)

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3)

model = LogisticRegression()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("The training accuracy of original dataset:{0: .1f}%".format(train_score * 100))
print("The test accuracy of original dataset:{0: .1f}%".format(test_score * 100))

#Without duplicate data
withoutduplicates = data.drop_duplicates(subset=None, keep='first', inplace=False)

X = withoutduplicates.iloc[:,:13].values
Y = withoutduplicates.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3)

model = LogisticRegression()
model.fit(X_train, Y_train)

uni_train_score = model.score(X_train, Y_train)
uni_test_score = model.score(X_test, Y_test)

print("The training accuracy of unique dataset:{0: .1f}%".format(uni_train_score * 100))
print("The test accuracy of unique dataset:{0: .1f}%".format(uni_test_score * 100))

#KNN
#test different k values
parameter_values = list(range(1, 21))
avg_scores = []
all_scores =[]
for n_neighbors in parameter_values:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn, x, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)
plt.plot(parameter_values, avg_scores, '-o')
plt.title('KNN accuracy using original date set')
plt.xlabel('K Values')
plt.ylabel('KNN Accuracy')
plt.show()

uni_avg_scores = []
uni_all_scores =[]
for n_neighbors in parameter_values:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn, X, Y, scoring='accuracy')
    uni_avg_scores.append(np.mean(scores))
    uni_all_scores.append(scores)
plt.plot(parameter_values, uni_avg_scores, '-o')
plt.title('KNN accuracy using unique date set')
plt.xlabel('K Values')
plt.ylabel('KNN Accuracy')
plt.show()

def test_autoNorm(x, y):
    #create a copy
    x_broken = np.array(x)
    # Divide the feature by 10 for every other column
    x_broken[:, ::2] /= 10
    knn = KNeighborsClassifier()
    # Calculate the accuracy of the original data
    original_scores = cross_val_score(knn, x, y, scoring='accuracy')
    print('The accuracy of the original data：{0: .1f}%'.format(np.mean(original_scores) * 100))
    # Calculate the accuracy of broken data
    broken_scores = cross_val_score(knn, x_broken, y, scoring='accuracy')
    print('The accuracy of broken data：{0: .1f}%'.format(np.mean(broken_scores) * 100))

    #normalize
    x_transformed = MinMaxScaler().fit_transform(x_broken)
    # Calculate the accuracy
    transformed_scores = cross_val_score(knn, x_transformed, y, scoring='accuracy')
    print('The accuracy of normalize data：{0: .1f}%'.format(np.mean(transformed_scores) * 100))

test_autoNorm(x, y)
test_autoNorm(X, Y)

#CART
datacolunm = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)
print('The test accuracy of CART in original dataset:{0: .1f}%'.format(dt.score(x_test, y_test)*100))
dot_data = export_graphviz(dt, out_file=None, feature_names= datacolunm,
                           class_names=["0", "1"], filled=True, rounded=True,
                           special_characters=True,precision=2)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("original_dt.png")

dt.fit(X_train, Y_train)
print('The test accuracy of CART in unique dataset:{0: .1f}%'.format(dt.score(X_test, Y_test)*100))
dot_data = export_graphviz(dt, out_file=None, feature_names= datacolunm,
                           class_names=["0", "1"], filled=True, rounded=True,
                           special_characters=True,precision=2)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("unique_dt.png")

#SVM
#normalize
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
X_std = scaler.fit_transform(X)
#split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=.2)

svc = svm.SVC(kernel='rbf', class_weight='balanced',)
c_range = np.logspace(-5, 15, 11, base=2)
gamma_range = np.logspace(-9, 3, 13, base=2)
#Using Grid search and cross validation, cv = 3.3
param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
clf = grid.fit(x_train, y_train)
# calculate the accuracy
score = grid.score(x_test, y_test)
print('The test accuracy of SVM in original dataset:{0: .1f}%'.format( score * 100))
uni_clf = grid.fit(X_train, Y_train)
# calculate the accuracy
uni_score = grid.score(X_test, Y_test)
print('The test accuracy of SVM in unique dataset:{0: .1f}%'.format( uni_score * 100))

