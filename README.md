# fmi-joke-recsys

## Introduction

This is a project for the 2025/26 course of "Deep Learning" under the Faculty of Mathematics and Informatics (FMI), part of Sofia University "St. Kliment Ohridski".

The project's goal is to compare different recommendation systems and approaches of generating recommendations of jokes. The jokes in question come from the [Jester Collaborative Filtering Dataset](https://goldberg.berkeley.edu/jester-data/), developed at the University of California, Berkeley.

In total, there are three types of recommendation systems compared to each other - Content-Based, Collaborative-Based and Deep Neural Recommender Systems. Content-Based RS contain a statistical method, a tf-idf/bow method. Collaborative-Based RS contain a SVD and a KNN based methods. The Deep Neural RS contains a custom Neural Net that predicts the rating a potential user might give to a new joke.

## Technologies

- jupyter
- matplotlib
- numpy
- pandas
- pillow
- scikit-learn
- scipy
- torch
