
# Hotels Reviews: topic modeling and review score predictor


## Problem
Nowadays people travel a lot. Travelers want to avoid hotels with poor service, remote location etc. and hope to find a cozy hotel to stay while traveling. Each person has its own taste of 'cozy'. Where are the perfect hotels to your taste located? What are travelers saying about that hotel? Do the topics cover what you care most about a living place? Are the travelers having positive experience or bad one concerning that? We propose to use hotel reviews data and perform text analysis and topic modeling using Naive Bayes and LDA. We also propose to build a machine learning model for predicting review scores from the features in the review data.


## Client
Travelers will certainly be interested in this project. They might spend lots of time searching/reading/evaluating hotels and the reviews. This project will save a vast amount of time for travelers. Hotel owners will be interested in knowing what customers are talking and caring about the hotels so as to improve service quality and maximize their business profit. 

## Data
The dataset is originally from [kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). It is about 515K hotel reviews data in Europe. It's a csv file containing information on hotel name, hotel address, review date, review scores, reviewers' nationality, positive/negative reviews' word count etc. 

## Method
Since we want to predict review scores, we need to build a predictive supervised machine learning model. We will first perform exploratory analysis, for instance, 
* visualize the distribution of review scores and the geographical distribution of hotels;
* apply the time series analysis to identify if there are seasonal trends in reviews; 
* find out if there is correlation between review score and reviewers' nationality.

Concerning the text data, we will first need to transform it into word vectors using models like bag-of-word model or beyond. We will also need to implement topic modelings using LDA. We can cluster hotels based on reviews or we also can cluster reviewers' based on reviews.


## Deliverables
A jupyter notebook containing the source codes and a presentation report will be shared through Github.
