## Overview
This lab uses machine learning to predict whether a student will **pass** or **fail** a course
based on data from two datasets: Student Math and Student Portuguese.
Using SVM (Support Vector Machines), the model classifies students based on their first
and second period grades.

## Problem
This app will explore whether we can predict if a student will pass their final exam
based on their first two grades in the course.

The binary classification problem is:
* 1 (pass) if the student's final grade (G3) is 10 or higher.
* 0 (fail) if the final grade is below 10.

The model uses the following two features:
* G1: First period grade
* G2: Second period grade

These two numerical features are used to train the SVM classifier, in attempt to draw
the best decision boundary to separate passing and failing students. 

## Dataset
**Source**: UCI Student Performance Dataset
* Student Math
* Student Portuguese

**Features Used**:
* G1: First period grade
* G2: Second period grade

**Target**:
* G3: Binary pass/fail based on final grade (10 or higher = pass, below 10 = fail)

https://archive.ics.uci.edu/dataset/320/student+performance
