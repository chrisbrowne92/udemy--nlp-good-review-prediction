"""
Objective of this code is to predict whether a review was 1 star or 5 stars, based on the text in the review
"""

# imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords as sw
from imblearn.over_sampling import RandomOverSampler

# read in data and split into test/training sets
reviews = pd.read_csv('data/yelp.csv')
# remove 2,3,4 start reviews
reviews = reviews[(reviews['stars'] == 1) | (reviews['stars'] == 5)]
# split into training / test sets first so that there is no knowledge of test set used during analysis/predictions
train, test = train_test_split(reviews, test_size=0.3, random_state=0, stratify=reviews['stars'])

'''
Explore training set
'''
print(train.info())
'''
Int64Index: 3268 entries, 5638 to 4869
Data columns (total 10 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   business_id  3268 non-null   object
 1   date         3268 non-null   object
 2   review_id    3268 non-null   object
 3   stars        3268 non-null   int64 
 4   text         3268 non-null   object
 5   type         3268 non-null   object
 6   user_id      3268 non-null   object
 7   cool         3268 non-null   int64 
 8   useful       3268 non-null   int64 
 9   funny        3268 non-null   int64 
 
 No missing data.
 '''

train['num_words'] = train['text'].apply(lambda x: len(x.split()))  # get number of words in each review
print(train.describe())
'''
             stars         cool       useful        funny    num_words
count  3268.000000  3268.000000  3268.000000  3268.000000  3268.000000
mean      4.282742     0.854651     1.392901     0.674113   119.732252
std       1.534697     2.405226     2.646706     1.973960   109.017271
min       1.000000     0.000000     0.000000     0.000000     1.000000
25%       5.000000     0.000000     0.000000     0.000000    47.000000
50%       5.000000     0.000000     1.000000     0.000000    89.000000
75%       5.000000     1.000000     2.000000     1.000000   160.250000
max       5.000000    77.000000    76.000000    39.000000   922.000000

Far more 5 star reviews and 1 star.
'''

# number of messages for each rating
print(train[['stars', 'text']].groupby('stars').count())
'''
stars      
1       586
5      2682

Confirmed.
'''

# histogram of number of words per review
ax = train['num_words'].hist(bins=25)
ax.set_xlabel('Review length (number of words)')
ax.set_ylabel('Frequency')
ax.figure.savefig('plots/01-review_length_distribution.png')
ax.figure.clf()
skew = train['num_words'].skew()
'''
Stong positive skew (2.4), most reviews are short.
'''

# how does message length change for different ratings?
g = sns.FacetGrid(train, col='stars', sharex=True)
g.map(plt.hist, 'num_words', bins=25, density=True)  # density plot to normalise bar height
g.savefig('plots/02-review_length_for_different_weightings-density_plot.png')
skew_ratings = train[['stars', 'num_words']].groupby('stars').agg(['skew', 'median'])
skew_ratings.round(3).to_clipboard()
'''
	  skew	  median
stars		
1	  2.069	  111
5	  2.349	  85

1 start reviews tend to be longer than 5-star reviews
'''

# heatmap of correlations
ax = sns.heatmap(train.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
ax.figure.savefig('plots/03-correlation_with_stars.png')
'''
Number of stars for a review has no correlation to the length of the text, 
or whether it was rated as 'cool', 'useful' or 'funny' by other users
'''

# get test and training sets
X_train = train['text']
X_test = test['text']
y_train = train['stars']
y_test = test['stars']

'''
Predictions using BOW vectorisation only
'''
ratings_predictor = Pipeline([('bow', CountVectorizer()),
                              ('model', MultinomialNB())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.85      0.65      0.74       151
           5       0.92      0.97      0.95       667
    accuracy                           0.91       818
   macro avg       0.89      0.81      0.84       818
weighted avg       0.91      0.91      0.91       818
'''

'''
Predictions using BOW vectorisation then apply tf-idf transform
'''
ratings_predictor = Pipeline([('bow', CountVectorizer()),
                              ('tf-idf', TfidfTransformer()),
                              ('model', MultinomialNB())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.00      0.00      0.00       151
           5       0.82      1.00      0.90       667
    accuracy                           0.82       818
   macro avg       0.41      0.50      0.45       818
weighted avg       0.66      0.82      0.73       818

It's less accurate. All the 1 star reviews were incorrectly predicted. Why...?
Is it because some of the important words are not in the training set? 
But then that defeats the purpose of having a train/test split. 
The training set should have enough vocab to be able to make a good prediction.
'''

'''
Try again, removing stopwords
'''
stopwords = sw.words('english')

'''
Without TF-IDF
'''
ratings_predictor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                              ('model', MultinomialNB())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.89      0.69      0.78       228
           5       0.93      0.98      0.96       998
    accuracy                           0.93      1226
   macro avg       0.91      0.83      0.87      1226
weighted avg       0.92      0.93      0.92      1226

Slight improvement, which would make sense as we have reduced the noise and therefore space for the model to overfit
'''

'''
With TF-IDF
'''
ratings_predictor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                              ('tf-idf', TfidfTransformer()),
                              ('model', MultinomialNB())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.00      0.00      0.00       228
           5       0.81      1.00      0.90       998
    accuracy                           0.81      1226
   macro avg       0.41      0.50      0.45      1226
weighted avg       0.66      0.81      0.73      1226

No improvement with TF-IDF.
'''

'''
Hypothesis: Words/phrases associated with bad/reviews "terrible", "good", "excellent" probably appear often. 
TF-IDF dow-weighs common words. Perhaps this makes it harder for the model to 'see' these words when TF-IDF is used, 
and so it's harder to make predictions about the tone of the review.

Test: Try training a random forest classifier to see which words are important for prediction of bad reviews, 
with and without TF-IDF

If hypothesis is true: Important words for TF-IDF pre-processing will be different (and unexpected) 
compared to the important words without TF-IDF
'''

# without TF-IDF
ratings_predictor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                              ('model', RandomForestClassifier())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.95      0.43      0.59       228
           5       0.88      0.99      0.94       998
    accuracy                           0.89      1226
   macro avg       0.92      0.71      0.76      1226
weighted avg       0.90      0.89      0.87      1226

Reduced F1-score but still very good (81%). 
Weakness in score comes from recall of bad reviews 
(i.e. many false negatives, wrongly classifying bad reviews as positive)

Which words does the model 'see' as important?
'''
word_to_idx = ratings_predictor.named_steps['preprocessor'].named_steps['bow'].vocabulary_  # get vocal list from bow
idx_to_word = dict((v, k) for k, v in word_to_idx.items())  # swap keys and values in dict
# get feature importance
important_words = pd.DataFrame(
    data=ratings_predictor.named_steps['model'].feature_importances_,
    columns=['importance'])
important_words.index = important_words.index.map(idx_to_word)  # apply mapping to index to get associated word
important_words = important_words.sort_values(by='importance', ascending=False).iloc[:20]  # sort and get top 20 words
print(important_words)
'''
            importance
horrible      0.014400
rude          0.013239
great         0.011034
awful         0.008815
worst         0.008308
bad           0.006641
disgusting    0.006640
gross         0.005957
love          0.005417
minutes       0.004972
money         0.004413
overpriced    0.004130
worse         0.004047
mediocre      0.004022
told          0.003999
poor          0.003932
best          0.003675
star          0.003643
manager       0.003563
closed        0.003392

These words seem reasonable to be predicting in bad/good reviews. How about when TF-IDF is used?
'''

# with TF-IDF
ratings_predictor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                              ('tf-idf', TfidfTransformer()),
                              ('model', RandomForestClassifier())]).fit(X_train, y_train)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
              precision    recall  f1-score   support
           1       0.94      0.35      0.51       228
           5       0.87      0.99      0.93       998
    accuracy                           0.87      1226
   macro avg       0.91      0.67      0.72      1226
weighted avg       0.88      0.87      0.85      1226

Reduced accuracy due to a drop in recall performance. Which are the important words now?
'''
word_to_idx = ratings_predictor.named_steps['preprocessor'].named_steps['bow'].vocabulary_  # get vocal list from bow
idx_to_word = dict((v, k) for k, v in word_to_idx.items())  # swap keys and values in dict
# get feature importance along with IDF
important_words = pd.DataFrame(
    data=zip(ratings_predictor.named_steps['model'].feature_importances_,
             pd.Series(ratings_predictor.named_steps['preprocessor'].named_steps['tf-idf'].idf_).rank(pct=True) * 100),
    columns=['importance', 'idf_percentile'])
important_words.index = important_words.index.map(idx_to_word)  # apply mapping to index to get associated word
important_words = important_words.sort_values(by='importance', ascending=False)  # sort and get top 20 words
print(important_words.iloc[:20])
'''
            importance  idf_percentile
horrible      0.019370        2.777950
great         0.011457        0.018644
rude          0.010433        3.731900
awful         0.010142        4.698279
told          0.007669        0.907340
worst         0.007587        3.570319
overpriced    0.006144        7.342614
minutes       0.006126        0.832764
gross         0.006001        5.829346
bad           0.005787        0.602821
poor          0.005511        4.418619
manager       0.005169        2.262134
worse         0.004773        4.912684
slow          0.004624        4.546020
star          0.004551        2.165807
disgusting    0.004449        7.699956
best          0.004381        0.080791
closed        0.004298        4.253931
mediocre      0.004255        7.090920
business      0.004226        1.165248

No obvious difference between the list of words. 
Both sets are what I'd associate with being able to judge review tone. 

idf_percentile shows the percentile for the word's IDF value, when ranked against all other words in the corpus.
A low percentile shows a lower IDF value - i.e. a less common word. The most common words are the negative ones.
-> implies bad review words are down-weighted and so perhaps are less meaningful to the model 
-> model struggles to positively identify bad reviews  

Try next: make the 1 star and 5 star reviews more even in number 
  - perhaps the model is being overfit to positive reviews in some way
'''


'''
Even sample sizes - try random oversampling to create even class groups
'''
X_train_os, y_train_os = RandomOverSampler().fit_sample(X_train.values.reshape(-1, 1), y_train)
ratings_predictor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                              ('tf-idf', TfidfTransformer()),
                              ('model', MultinomialNB())]).fit(np.squeeze(X_train_os), y_train_os)
y_pred = ratings_predictor.predict(X_test)
results = classification_report(y_test, y_pred)
print(results)
'''
           precision    recall  f1-score   support
           1       0.70      0.91      0.79       228
           5       0.98      0.91      0.94       998
    accuracy                           0.91      1226
   macro avg       0.84      0.91      0.87      1226
weighted avg       0.93      0.91      0.92      1226

Massive improvement, compared to F1-score without oversample, which was:
              precision    recall  f1-score   support
           1       0.00      0.00      0.00       228
           5       0.81      1.00      0.90       998
    accuracy                           0.81      1226
   macro avg       0.41      0.50      0.45      1226
weighted avg       0.66      0.81      0.73      1226

Conclusion: 
oor performance was due to imbalanced class sizes. 
Using random oversampling to even the class sizes in the training set improved model F1 score from 0.73 -> 0.92. 
Improvement was seen in all metrics within the F1 score, but mainly in the recognition performance of the bad 
reviews (which originally had lower representation in the sample). 
The model went from having 0% on bad reviews to 0.7 recall & 0.91.
'''