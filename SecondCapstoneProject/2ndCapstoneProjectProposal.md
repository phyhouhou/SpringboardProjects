
# Hotels Reviews: topic modeling and review score predictor


## Problem
Nowadays, people travel very often for business or for holidays. Travelers want to select hotels with clean rooms, high-quality service, convenient location etc. In a word, people hope to find a cozy and cost-effective hotel to stay while traveling. A massive amount of reviews are being posted online and people are influenced by online reviews in making their decisions. Each person has its own taste of 'cozy'. Where are the perfect hotels to your taste located? What are other travelers saying about that hotel? Are previous travelers having positive experience or bad one concerning your needs? We propose to investigate hotel reviews data and perform text analysis and topic modeling using Naive Bayes and LDA. We also propose to build a machine learning model for predicting review scores from the features we have in the data.


## Client
Travelers will certainly be interested in this project. They might spend lots of time searching/reading/evaluating hotels and the reviews. This project will save a vast amount of time for travelers. Hotel owners are eager to know what customers are talking and especially caring about the hotels. This project can help them improve service quality and maximize their business profit. Other potential clients include travel service agencies and housing agencies etc. 


## Data
The dataset is originally from [kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). It is about 515K hotel reviews data in Europe. It's a csv file containing information on hotel name, hotel address, review date, review scores, reviewers' nationality, positive/negative reviews' word count etc. 

The data can be enriched by adding hotel features acquired from, for instance, trip advisor etc.

## Method
We will conduct data wrangling and cleaning. Since there are both numerical and textual features, we will perform exploratory analysis correspondingly.

Concerning the numeric data, we will 
* visualize the distribution of review scores and the geographical distribution of hotels;
* apply the time series analysis to identify if there are seasonal trends in review scores; 
* find out if there is correlation between review score and reviewers' nationality.

Concerning the textual review data, we will:
* transform text data into word vectors using models like bag-of-word model or beyond; 
* figure out strong predictive words;
* implement LDA type topic modelings on both the positive and negative reviews;

After the EDA and feature engineering, we will build a predictive supervised machine learning model to predict ratings based on all the features we have. 

## Deliverables
A jupyter notebook containing the source codes and a presentation report will be shared through Github.


