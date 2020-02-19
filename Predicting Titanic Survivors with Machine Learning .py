#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# Get data overview

# In[2]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.head())


# General info: total number, data types.
# Could find missing value(age and cabin)

# In[3]:


print(train_data.info())


# Get more info

# In[4]:


print(train_data.describe()) 


# In[6]:


train_data.describe(include=['O']) 


# Drop unrelated feature, and features with too many missing value

# In[7]:


train_data = train_data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) 
test_data = test_data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) 
train_data.head() # check everything looks okay


# Visualizing survived and failed people

# In[8]:


train_data.Survived.value_counts().plot(kind='bar')
plt.title("Distribution of Survival, (1 = Survived)")


# In[9]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[10]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Sex', bins=3)


# In[11]:


train_data.Pclass.value_counts().plot(kind="barh")
plt.title("Class Distribution")


# In[12]:


pclass_survived = train_data[train_data['Survived']==1]['Pclass'].value_counts()
pclass_dead = train_data[train_data['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([pclass_survived,pclass_dead])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(10,8))


# In[13]:


train_data.Embarked.value_counts().plot(kind='bar')
plt.title("Passengers per boarding location")


# In[14]:


survived = train_data[train_data['Survived']==1]['Embarked'].value_counts()
dead = train_data[train_data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=False, figsize=(8,6))
plt.title("Survival and Death in Different ports")


# In[15]:


survived_0 = train_data[train_data['Survived'] == 0]["Fare"].mean()
survived_1 = train_data[train_data['Survived'] == 1]["Fare"].mean()
xs  = [survived_0, survived_1]
ys = ['Dead','Survived']
plt.bar(ys, xs, 0.6, align='center',color = 'green')
plt.xlabel('Outcome')
plt.ylabel('Mean Fare')
plt.show()


# In[16]:


def wrangle(dataset):
    # sex {male, female} to {0, 1}
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # embarked {S, C, Q} => 3 binary variables
    embarked_separate_port = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    dataset = pd.concat([dataset, embarked_separate_port], axis=1)
    return dataset.drop('Embarked', axis=1)
 
train_data = wrangle(train_data)
test_data = wrangle(test_data)
train_data.head()


# In[17]:


corr = train_data.corr()
print(corr)


# In[18]:


corr = train_data.corr()
print(corr)


# In[19]:


sns.heatmap(np.abs(corr),          # use absolute values
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[20]:


guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_data = train_data[(train_data['Sex'] == i) & (train_data['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_data.median()
        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
 
def wrangle_age(dataset):
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
    return dataset
 
train_data = wrangle_age(train_data)
test_data = wrangle_age(test_data)


# In[21]:


#train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
#train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1


# In[22]:


print(train_data.info())


# In[23]:


print(test_data.info())


# In[24]:


mean_fare = 32
test_data['Fare'] = test_data['Fare'].fillna(32)


# In[25]:


print(test_data.info())


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Split validation and training data

# In[27]:


X_train = train_data.drop("Survived", axis=1)[:800]
print(X_train.info())
Y_train = train_data["Survived"][:800]
X_crossValidation = train_data.drop("Survived", axis=1)[800:]
Y_crossValidation = train_data["Survived"][800:]
X_test = test_data
print(test_data.info())


# logistic regression

# In[28]:


model_logistic = LogisticRegression(max_iter=4000)
model_logistic.fit(X_train, Y_train)
train_accuracy = round(model_logistic.score(X_train, Y_train) * 100, 2)
validation_accuracy = round(model_logistic.score(X_crossValidation, Y_crossValidation) * 100, 2)
Y_predL = model_logistic.predict(X_test)
print(train_accuracy)
print(validation_accuracy)


# In[29]:


svc = SVC()
svc.fit(X_train, Y_train)
train_accuracy = round(svc.score(X_train, Y_train) * 100, 2)
validation_accuracy = round(svc.score(X_crossValidation, Y_crossValidation) * 100, 2)
Y_predS = svc.predict(X_test)
print(train_accuracy)
print(validation_accuracy)


# In[30]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
train_accuracy = round(knn.score(X_train, Y_train) * 100, 2)
validation_accuracy = round(knn.score(X_crossValidation, Y_crossValidation) * 100, 2)
Y_predK = knn.predict(X_test)
print(train_accuracy)
print(validation_accuracy)


# In[31]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
train_accuracy = round(random_forest.score(X_train, Y_train) * 100, 2)
validation_accuracy = round(random_forest.score(X_crossValidation, Y_crossValidation) * 100, 2)
Y_predR = random_forest.predict(X_test)
print(train_accuracy)
print(validation_accuracy)


# In[34]:


# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV


# In[35]:


# parameter_grid = {
#              'max_depth' : [4, 6, 8],
#              'n_estimators': [10, 50,100],
#              'max_features': ['sqrt', 'auto', 'log2'],
#              'min_samples_split': [0.001,0.003,0.01],
#              'min_samples_leaf': [1, 3, 10],
#              'bootstrap': [True,False],
#              }
# forest = RandomForestClassifier()
# cross_validation = StratifiedKFold(Y_train, n_folds=5)
 
# grid_search = GridSearchCV(forest,
#                            scoring='accuracy',
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
 
# grid_search.fit(X_train, Y_train)
# model = grid_search
# parameters = grid_search.best_params_
 
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:





# In[ ]:




