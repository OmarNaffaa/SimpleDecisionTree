# Attempt to visualize reduced Self-Healing Data
# tutorial resource: https:///www.datacamp.com/community/tutorials/decision-tree-classification-python
# ----------------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from sklearn.tree import DecisionTreeClassifier         # import decision tree classifier
from sklearn.model_selection import train_test_split    # import train_test_split function
from sklearn import metrics                             # import scikit-learn metrics module for accuracy calculation
# Visualization imports
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
# Import OS to set path to use Graphviz
import os     
os.environ["PATH"] += os.pathsep + 'C:\\Users\\Omar\\Downloads\\graphviz-2.38\\release\\bin'

# Load data in
col_names = ['Time', 'BS_ID', 'BS_LOC_X', 'BS_LOC_Y', 'ANT_ID', 'TRX_ID', 'TRX_X', 'TRX_Y', 'TRX_ANGLE', 'UE_ID', 'UE_LOC_X', 'UE_LOC_Y', 'MAX_DR', 'DEMAND_DR', 'REAL_DR', 'BS_Tran_SNR', 'UE_Rec_SNR', 'output']
selfHealingDB = pd.read_csv("SmallDataSheet.csv", header=None, names=col_names, index_col=0)

selfHealingDB.head()

# print out data sheet for diagnostics
print(selfHealingDB)

# Divide dataset into features (variables we are measuring) and target (variable we are trying to model / forcast)
# Scenario: using 3 types of data rates with fake binary output value
featureCols = ['MAX_DR', 'DEMAND_DR', 'REAL_DR']
targetVal = selfHealingDB.output                       # target value must be 0 or 1 only

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
export_graphviz(clf, out_file=dotData, filled=True, rounded=True, special_characters=True, feature_names = featureCols, class_names=['0','1'])

print("\nGraphing using pydotplus library...")
graph = pydotplus.graph_from_dot_data(dotData.getvalue())
graph.write_png('BigTheta.png')

print("Printing now...")
Image(graph.create_png())
