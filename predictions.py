"""
Objective of this code is to predict whether a review was 1 star or 5 stars, based on the text in the review
"""

# imports
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

# read in data and split into test/training sets
reviews = pd.read_csv('data/yelp.csv')
# remove 2,3,4 start reviews
reviews = reviews[(reviews['stars'] == 1) | (reviews['stars'] == 5)]
# split into training / test sets first so that there is no knowledge of test set used during analysis/predictions
train, test = train_test_split(reviews, test_size=0.3, random_state=0)

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
y_map = {1: 0, 5: 1}
X_train = train['text']
X_test = test['text']
y_train = train['stars']
y_test = test['stars']

'''
Predictions using BOW vectorisation only
'''
preprocessor = Pipeline([('bow', CountVectorizer())])  # only bag of words
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', MultinomialNB())])\
    .fit(X_train, y_train)
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
preprocessor = Pipeline([('bow', CountVectorizer()),
                         ('tf-idf', TfidfTransformer())])
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', MultinomialNB())])\
    .fit(X_train, y_train)
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
preprocessor = Pipeline([('bow', CountVectorizer(stop_words=stopwords))])  # only bag of words
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', MultinomialNB())])\
    .fit(X_train, y_train)
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
preprocessor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                         ('tf-idf', TfidfTransformer())])
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', MultinomialNB())])\
    .fit(X_train, y_train)
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
preprocessor = Pipeline([('bow', CountVectorizer(stop_words=stopwords))])  # only bag of words
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', RandomForestClassifier())])\
    .fit(X_train, y_train)
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
Weakness in score comes from recall of bad reviews (i.e. many false negatives)

Which words does the model 'see' as important?
'''
word_to_idx = ratings_predictor.named_steps['preprocessor'].named_steps['bow'].vocabulary_  # get vocal list from bow
idx_to_word = dict((v, k) for k, v in word_to_idx.items())  # swap keys and values in dict
important_words = pd.Series(ratings_predictor.named_steps['model'].feature_importances_)  # get feature importance
important_words.index = important_words.index.map(idx_to_word)  # apply mapping to index to get assiciated word
important_words = important_words.sort_values(ascending=False).iloc[:20]  # sort and get top 20 words
print(important_words)
'''
rude          0.015203
horrible      0.014137
great         0.010104
awful         0.009589
worst         0.008282
worse         0.006258
minutes       0.006086
manager       0.005507
disgusting    0.005433
overpriced    0.005409
bad           0.005251
slow          0.005016
mediocre      0.004823
love          0.004681
gross         0.004679
poor          0.004620
asked         0.004498
terrible      0.004486
told          0.004379
closed        0.004285

These words seem reasonable to be predicting in bad/good reviews. How about when TF-IDF is used?
'''

# with TF-IDF
preprocessor = Pipeline([('bow', CountVectorizer(stop_words=stopwords)),
                         ('tf-idf', TfidfTransformer())])
ratings_predictor = Pipeline([('preprocessor', preprocessor),
                              ('model', RandomForestClassifier())])\
    .fit(X_train, y_train)
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
important_words = pd.Series(ratings_predictor.named_steps['model'].feature_importances_)  # get feature importance
important_words.index = important_words.index.map(idx_to_word)  # apply mapping to index to get assiciated word
important_words = important_words.sort_values(ascending=False).iloc[:20]  # sort and get top 20 words
print(important_words)
'''
rude          0.014839
horrible      0.014500
great         0.012416
told          0.007915
awful         0.007900
worst         0.007611
gross         0.006825
worse         0.006657
bad           0.006645
poor          0.006084
best          0.005340
manager       0.005022
business      0.004881
mediocre      0.004843
sorry         0.004425
nothing       0.004282
slow          0.004204
avoid         0.004178
said          0.004175
disgusting    0.004091

No obvious difference between the list of words. 
Both sets are what I'd associate with being able to judge review tone. 

Is there a wider range of 'bad' words used in bad reviews, vs 'good' words used in good reviews?

LOOK AT IDF FOR THE LISTS OF WORDS ABOVE. ARE THE BAD WORDS LOWER WEIGHTED?
'''