import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from word2number import w2n
pd.options.mode.chained_assignment = None

df = pd.read_csv('interview.csv')

#clean the data by filling empty values
median_test_score_outof_10 = math.floor(df.test_score_outof_10.median())
df.test_score_outof_10 = df.test_score_outof_10.fillna(median_test_score_outof_10)

#converting words to numbers
df.experience = df.experience.fillna('zero')
a = 0
for e in df.experience:
    df.experience.iloc[[a]] = w2n.word_to_num(e)
    a += 1

#create linear regression object
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score_outof_10', 'interview_score_outof_10']], df.salary)

#input new csv file to predict
d = pd.read_csv('interview_results.csv')

#clean the data by filling empty values
median_test_score_outof_10 = math.floor(d.test_score_outof_10.median())
d.test_score_outof_10 = d.test_score_outof_10.fillna(median_test_score_outof_10)

#converting words to numbers
d.experience = d.experience.fillna('zero')
b = 0
for e in d.experience:
    d.experience.iloc[[b]] = w2n.word_to_num(e)
    b += 1

#predict salary
sal = reg.predict(d)

#create new csv file output
d['salary'] = sal
d.to_csv('salary_prediction.csv', index=False)
