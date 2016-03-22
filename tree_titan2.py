import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
d_data = data[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]
clean_data = d_data.dropna()
clean_data['Sex'] = clean_data['Sex'].replace(('male', 'female'), (0, 1))
survived_arr = clean_data['Survived'].as_matrix()
x_df = clean_data[['Pclass', 'Fare', 'Age', 'Sex']]

#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state = 241)
clf.fit(x_df, survived_arr)
I = clf.feature_importances_

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=('Pclass', 'Fare', 'Age', 'Sex'),
                         class_names=('Survived'),
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("titan.pdf")

from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=('Pclass', 'Fare', 'Age', 'Sex'),
                         class_names=('Survived'),
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


#from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("test.pdf")
print I