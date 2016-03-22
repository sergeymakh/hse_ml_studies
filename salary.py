import pandas as pd
train = pd.read_csv("salary-train.csv")
test = pd.read_csv("salary-test-mini.csv")
train['FullDescription'] = train['FullDescription'].str.lower()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
#print data.columns.values
#print train.head()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5)
tfid_train = vectorizer.fit_transform(train['FullDescription'])
tfid_test = vectorizer.transform(test['FullDescription'])
#print tfid

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

#print X_train_categ
#print X_test_categ

print X_train_categ.size
print tfid_train.size
print len(train['FullDescription'])
from scipy.sparse import hstack
train_matrix = hstack([tfid_train, X_train_categ])
test_matrix = hstack([tfid_test, X_test_categ])
#print train_matrix[59999]
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(train_matrix, train['SalaryNormalized'])
predicted_salary = clf.predict(test_matrix)
print round(predicted_salary[0], 2), round(predicted_salary[1], 2)