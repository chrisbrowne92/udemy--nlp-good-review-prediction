Using Natural Language Processing to predict the tone of a review - is it 1 stars or 5 stars based on the vocabulary used?

One of the portfolio projects for the Udemy ML course I am doing.

At the end of the exercise it was found that TF-IDF reduced model performance, and that the overall performance 
was being reduced by poor classification of bad reviews. 
In fact, with TF-IDF the model didn't classify a single bad review correctly. 
The course module ended there with the message that TF-IDF didn't work that time but no real explanation. 

I wanted to understand why and so did some investigation. 

After some investigation I found that the number of good:bad reviews was approx an 80:20 ratio, and felt it could be 
that the models were being over-fit to the good reviews in the training set.
In order to address this, I tried randomly oversampling the training set, in order to build even-sized class groups.
This was very successful. The performace of the original model, without oversampling was:
```
              precision    recall  f1-score   support
           1       0.00      0.00      0.00       228
           5       0.81      1.00      0.90       998
    accuracy                           0.81      1226
   macro avg       0.41      0.50      0.45      1226
weighted avg       0.66      0.81      0.73      1226
```

With oversampling the performance was:
```
              precision    recall  f1-score   support
           1       0.70      0.91      0.79       228
           5       0.98      0.91      0.94       998
    accuracy                           0.91      1226
   macro avg       0.84      0.91      0.87      1226
weighted avg       0.93      0.91      0.92      1226
```

A massive improvement was seen in the classification of bad reviews (F1: 0.00 -> 0.79), with no reduction in 
performance for good reviews - in fact there was a modest increase in performance (F1: 0.90 -> 0.94).

Overall F1 score increased from 0.73 -> 0.92 