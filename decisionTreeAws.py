%pyspark
import numpy as np
import pandas as pd
import s3fs
import graphviz
from sklearn.tree import DecisionTreeClassifier         # import decision tree classifier
from sklearn.model_selection import train_test_split    # import train_test_split function
from sklearn import metrics                             # import scikit-learn metrics module for accuracy calculation

# Visualization imports
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus

#Load data in using S3FS
#s3 = s3fs.S3FileSystem(anon=True)
#with s3.open('s3://cppbigdata2/status_test1/part-00000-7a95ea77-de25-4b73-a170-934410551b04-c000.csv', 'rb') as f:
    #selfHealingDB = pd.read_csv(f, header=0)

col_names = ['Time', 'MAX_DR', 'DEMAND_DR', 'REAL_DR', 'status']
selfHealingDB = pd.read_csv('s3://cppbigdata1/status_test_coalesce/sim10_100_500.csv/part-00000-b891c7e3-32e2-4c3d-a036-5e2fe2687e83-c000.csv', header=None, names=col_names, index_col=0)

# print out data sheet for diagnostics
print(selfHealingDB)

# Divide dataset into features (variables we are measuring) and target (variable we are trying to model / forcast)
# Scenario: using 3 types of data rates with fake binary output value
featureCols = ['MAX_DR', 'DEMAND_DR', 'REAL_DR']
targetVal = selfHealingDB.status                   # target value must be discrete 

X = selfHealingDB[featureCols]
y = targetVal

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% testing

# Create decision tree classifier object
clf = DecisionTreeClassifier(max_depth=5) # note: arguments are optional (used for pruning)

# Train decision tree classifier
clf = clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Evaluate model by printing the model accuracy (how often the classifier is correct)
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
print()

# attempting to visualize the data
print("Beginning Visualization...")
dotData = StringIO()

print("\nExporting Graphviz...")
export_graphviz(clf, out_file=dotData, filled=True, rounded=True, special_characters=True, feature_names = featureCols, class_names=['failed', 'congested', 'healthy'])

print("\nGraphing using pydotplus library...")
graph = pydotplus.graph_from_dot_data(dotData.getvalue())
graph.write_png('Tree.png')

print()
print("Printing now...")
print()

Image(graph.create_png())
