
# Hotels Reviews: topic modeling and review score predictor


## Problem
Nowadays, people travel very often for business or for holidays. Travelers want to select hotels with clean rooms, high-quality service, convenient location etc. In a word, people hope to find a cozy and cost-effective hotel to stay while traveling. A massive amount of reviews are being posted online. People are effectively influenced by online reviews in making their decisions. Each person has its own taste of 'cozy'. Where are the perfect hotels to your taste located? What are other travelers saying about that hotel? Do the reviews cover what you care most about a living place? Are previous travelers having positive experience or bad one concerning your needs and pain-points? We propose to use hotel reviews data and perform text analysis and topic modeling using Naive Bayes and LDA. We also propose to build a machine learning model for predicting review scores from the features in the review data.


## Client
Travelers will certainly be interested in this project. They might spend lots of time searching/reading/evaluating hotels and the reviews. This project will save a vast amount of time for travelers. Hotel owners are eager to know what customers are talking and especially caring about the hotels. This project can help them improve service quality and maximize their business profit. Other potential clients include travel service agencies and housing agencies etc. 


## Data
The dataset is originally from [kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). It is about 515K hotel reviews data in Europe. It's a csv file containing information on hotel name, hotel address, review date, review scores, reviewers' nationality, positive/negative reviews' word count etc. 

## Method
Since we want to predict review scores, we need to build a predictive supervised machine learning model. We will first perform exploratory analysis, for instance, 
* visualize the distribution of review scores and the geographical distribution of hotels;
* apply the time series analysis to identify if there are seasonal trends in reviews; 
* find out if there is correlation between review score and reviewers' nationality.

Concerning the text data, we will first need to transform it into word vectors using models like bag-of-word model or beyond. We can figure out strong predictive words. We will also need to implement topic modelings using LDA. We can cluster hotels based on reviews or we also can cluster reviewers' based on reviews.


## Deliverables
A jupyter notebook containing the source codes and a presentation report will be shared through Github.
