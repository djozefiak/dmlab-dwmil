## Data Mining Lab 2020 @ Leibniz Universität Hannover

### Dynamic Weighted Majority for Incremental Learning of Imbalanced Data Streams with Concept Drift

### Introduction

* **concept drifts** occurring in data streams will jeopardize the accuracy and stability
* if the data stream is **imbalanced**, it will be even more challenging to detect and handle the concept drift
* these two problems have been intensively addressed separately
* they have yet to be well studied when they occur together

### Method

* chunk-based incremental learning method
* deals with data streams with concept drift and class imbalance problem
* creates a base classifier for each chunk
* weighs them by their performance tested on the current chunk
* a classifier trained recently or on a similar concept will receive a high weight

### Merits

1. can keep stable for non-drifted streams and quickly adapt to the new concept
2. is totally incremental, no previous data needs to be stored
3. keeps a limited number of classifiers to ensure high efficiency
4. is simple and needs only one threshold parameter

### Results

According to the authors, DWMIL should perform better on both synthetic and real data sets with concept drift than the state-of-the-art competitors, with less computational cost.

### Wiki

[Wiki-pages](https://github.com/djozefiak/dmlab-dwmil/wiki) contain an analysis of a data set and the implementation description.

### Reference

\[Yang Lu, Yiu-ming Cheung and Yuan Yan Tang\] [Dynamic Weighted Majority for Incremental Learning of Imbalanced Data Streams with Concept Drift](https://doi.org/10.24963/ijcai.2017/333). In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, pages 2393-2399. IJCAI-17, 2017.
