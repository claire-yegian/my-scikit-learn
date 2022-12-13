##############################################
# Programmer: Claire Yegian
# Class: CPSC 322-01, Fall 2022
# Programming Assignment #7
# 11/21/22
# Description: tests myclassifiers.py
##############################################
import numpy as np
from scipy import stats

import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MySimpleLinearRegressor, \
    MySimpleLinearRegressionClassifier,MyKNeighborsClassifier, \
    MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

def high_low_discretizer(value):
    if value <= 100:
        return "low"
    return "high"

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    # TDD: test driven development
    # write unit tests before writing units themselves
    # fully understand how to write the unit if you fully understand how to know
    # it is correct
    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer)
    lin_clf.fit(X_train, y_train) # "fits" slope (m) and intercept (b)

    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    assert np.isclose(lin_clf.regressor.slope, slope_solution)
    assert np.isclose(lin_clf.regressor.intercept, intercept_solution)

def test_simple_linear_regression_classifier_predict():
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer,
        MySimpleLinearRegressor(2, 10)) # y = 2x + 10

    X_test = [[78], [12], [7]]
    y_predicted_solution = ["high", "low", "low"]
    y_predicted = lin_clf.predict(X_test)
    assert y_predicted == y_predicted_solution

    lin_clf2 = MySimpleLinearRegressionClassifier(high_low_discretizer,
        MySimpleLinearRegressor(8, 3)) # y = 8x + 3

    X_test2 = [[51], [1], [91], [7]]
    y_predicted_solution2 = ["high", "low", "high", "low"]
    y_predicted2 = lin_clf2.predict(X_test2)
    assert y_predicted2 == y_predicted_solution2

def test_kneighbors_classifier_kneighbors():
    # TEST AGAINST IN-CLASS EXAMPLE 1
    X_train_class_example1 = [[1, 1], [1, 0], [1/3, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[1/3, 1]]

    knn_clf1 = MyKNeighborsClassifier(3)
    knn_clf1.fit(X_train_class_example1, y_train_class_example1)
    distances1 = [[2/3, 1, 1.054093]]
    indicies1 = [[0, 2, 3]]

    y_predicted_dist1, y_predicted_idxs1 = knn_clf1.kneighbors(X_test1)
    assert np.allclose(y_predicted_dist1, distances1)
    assert np.allclose(y_predicted_idxs1, indicies1)

    # TEST AGAINST IN-CLASS EXAMPLE 2
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[2, 3]]

    knn_clf2 = MyKNeighborsClassifier(3)
    knn_clf2.fit(X_train_class_example2, y_train_class_example2)
    distances2 = [1.41421356, 1.41421356, 2.        ]
    indicies2 = [0, 4, 6]

    y_predicted_dist2, y_predicted_idxs2 = knn_clf2.kneighbors(X_test2)
    assert np.allclose(y_predicted_dist2, distances2)
    assert np.allclose(y_predicted_idxs2, indicies2)

    # TEST AGAINST BRAMER EXAMPLE
    X_train_bramer_example = [[0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer = [[9.1, 11.0]]

    knn_clf_bramer = MyKNeighborsClassifier(5)
    knn_clf_bramer.fit(X_train_bramer_example, y_train_bramer_example)
    distances_bramer = [[2.802, 1.237, 0.608, 2.202, 2.915]]
    indicies_bramer = [[4, 5, 6, 7, 8]]

    y_predicted_dist_bramer, y_predicted_idxs_bramer = knn_clf_bramer.kneighbors(X_test_bramer)
    # the returned distances have too many digits after the decimal for all close to work, so I had
    # to round them
    for i in range(len(y_predicted_dist_bramer)):
        y_predicted_dist_bramer[i] = np.round(y_predicted_dist_bramer, 3)
    assert np.allclose(y_predicted_dist_bramer, distances_bramer)
    assert np.allclose(y_predicted_idxs_bramer, indicies_bramer)

def test_kneighbors_classifier_predict():
    # TEST AGAINST IN-CLASS EXAMPLE 1
    X_train_class_example1 = [[1, 1], [1, 0], [1/3, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[1/3, 1]]

    knn_clf1 = MyKNeighborsClassifier(3)
    knn_clf1.fit(X_train_class_example1, y_train_class_example1)

    predictions1 = ["good"]
    y_predicted1 = knn_clf1.predict(X_test1)
    assert y_predicted1 == predictions1

    # TEST AGAINST IN-CLASS EXAMPLE 2
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[2, 3]]

    knn_clf2 = MyKNeighborsClassifier(3)
    knn_clf2.fit(X_train_class_example2, y_train_class_example2)
    predictions2 = ["yes"]

    y_predicted2 = knn_clf2.predict(X_test2)
    assert y_predicted2 == predictions2

    # TEST AGAINST BRAMER EXAMPLE
    X_train_bramer_example = [[0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer = [[9.1, 11.0]]

    knn_clf_bramer = MyKNeighborsClassifier(5)
    knn_clf_bramer.fit(X_train_bramer_example, y_train_bramer_example)

    predictions_bramer = ['+']
    y_predicted_bramer = knn_clf_bramer.predict(X_test_bramer)
    assert y_predicted_bramer == predictions_bramer

def test_dummy_classifier_fit():
    X_train1 = [[value, value*2] for value in range(100)]
    y_train1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))

    dummy_clf1 = MyDummyClassifier()
    dummy_clf1.fit(X_train1, y_train1)
    fit1 = "yes"
    y_model1 = dummy_clf1.most_common_label
    assert fit1 == y_model1

    X_train2 = [[value-33, value**(1/3)] for value in range(100)]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))

    dummy_clf2 = MyDummyClassifier()
    dummy_clf2.fit(X_train2, y_train2)
    fit2 = "no"
    y_model2 = dummy_clf2.most_common_label
    assert fit2 == y_model2

    X_train3 = [[2/(value + 1), value+12] for value in range(100)]
    y_train3 = list(np.random.choice(["fraud", "not fraud"], 100, replace=True, p=[0.01, 0.99]))

    dummy_clf3 = MyDummyClassifier()
    dummy_clf3.fit(X_train3, y_train3)
    fit3 = "not fraud"
    y_model3 = dummy_clf3.most_common_label
    assert fit3 == y_model3

def test_dummy_classifier_predict():
    X_train1 = [[value, value*2] for value in range(100)]
    y_train1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_test1 = [[101, 202]]

    dummy_clf1 = MyDummyClassifier()
    dummy_clf1.fit(X_train1, y_train1)
    predictions1 = ["yes"]
    y_predictions1 = dummy_clf1.predict(X_test1)
    assert predictions1 == y_predictions1

    X_train2 = [[value-33, value**(1/3)] for value in range(100)]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_test2 = [[72, 4], [81, 7]]

    dummy_clf2 = MyDummyClassifier()
    dummy_clf2.fit(X_train2, y_train2)
    predictions2 = ["no","no"]
    y_predictions2 = dummy_clf2.predict(X_test2)
    assert predictions2 == y_predictions2

    X_train3 = [[2/(value + 1), value+12] for value in range(100)]
    y_train3 = list(np.random.choice(["fraud", "not fraud"], 100, replace=True, p=[0.01, 0.99]))
    X_test3 = [[1, 33]]

    dummy_clf3 = MyDummyClassifier()
    dummy_clf3.fit(X_train3, y_train3)
    predictions3 = ["not fraud"]
    y_predictions3 = dummy_clf3.predict(X_test3)
    assert predictions3 == y_predictions3

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [ # header = ["att1", "att2"]
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    priors_sol_inclass_ex = {"yes": 5/8,
                             "no":  3/8}
    posteriors_sol_inclass_ex = {"yes": {
                                    "att1=1": 4/5,
                                    "att1=2": 1/5,
                                    "att2=5": 2/5,
                                    "att2=6": 3/5},
                                 "no": {
                                    "att1=1": 2/3,
                                    "att1=2": 1/3,
                                    "att2=5": 2/3,
                                    "att2=6": 1/3}}
    nb_clf_inclass_ex = MyNaiveBayesClassifier()
    nb_clf_inclass_ex.fit(X_train_inclass_example, y_train_inclass_example)
    assert nb_clf_inclass_ex.priors == priors_sol_inclass_ex
    assert nb_clf_inclass_ex.posteriors == posteriors_sol_inclass_ex

    # RQ5 (fake) iPhone purchases dataset
    X_train_iphone = [ # header = ["standing", "job_status", "credit_rating", "buys_iphone"]
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]
    priors_sol_iphone = {"yes": 10/15, #dictionary on buys_iphone classification
                         "no":  5/15}
    posteriors_sol_iphone = {"yes": {
                                "att1=1": 2/10, #att1 = standing
                                "att1=2": 8/10,
                                "att2=1": 3/10, #att2 = job_status
                                "att2=2": 4/10,
                                "att2=3": 3/10,
                                "att3=fair": 7/10, #att3 = credit_rating
                                "att3=excellent": 3/10},
                             "no": {
                                "att1=1": 3/5,
                                "att1=2": 2/5,
                                "att2=1": 1/5,
                                "att2=2": 2/5,
                                "att2=3": 2/5,
                                "att3=fair": 2/5,
                                "att3=excellent": 3/5}}
    nb_clf_iphone = MyNaiveBayesClassifier()
    nb_clf_iphone.fit(X_train_iphone, y_train_iphone)
    assert nb_clf_iphone.priors == priors_sol_iphone
    assert nb_clf_iphone.posteriors == posteriors_sol_iphone

    # Bramer 3.2 train dataset
    X_train_bramer = [ # header = ["day", "season", "wind", "rain", "class"]
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    priors_sol_bramer = {"on time": 14/20,
                         "late":  2/20,
                         "very late": 3/20,
                         "cancelled": 1/20}
    posteriors_sol_bramer = {"on time": {
                                "att1=weekday": 9/14, #att1 = day
                                "att1=saturday": 2/14,
                                "att1=sunday": 1/14,
                                "att1=holiday": 2/14,
                                "att2=spring": 4/14, #att2 = season
                                "att2=summer": 6/14,
                                "att2=autumn": 2/14,
                                "att2=winter": 2/14,
                                "att3=none": 5/14, #att3 = wind
                                "att3=high": 4/14,
                                "att3=normal": 5/14,
                                "att4=none": 5/14, #att4 = rain
                                "att4=slight": 8/14,
                                "att4=heavy": 1/14},
                             "late": {
                                "att1=weekday": 1/2,
                                "att1=saturday": 1/2,
                                "att1=sunday": 0/2,
                                "att1=holiday": 0/2,
                                "att2=spring": 0/2,
                                "att2=summer": 0/2,
                                "att2=autumn": 0/2,
                                "att2=winter": 2/2,
                                "att3=none": 0/2,
                                "att3=high": 1/2,
                                "att3=normal": 1/2,
                                "att4=none": 1/2,
                                "att4=slight": 0/2,
                                "att4=heavy": 1/2},
                             "very late": {
                                "att1=weekday": 3/3,
                                "att1=saturday": 0/3,
                                "att1=sunday": 0/3,
                                "att1=holiday": 0/3,
                                "att2=spring": 0/3,
                                "att2=summer": 0/3,
                                "att2=autumn": 1/3,
                                "att2=winter": 2/3,
                                "att3=none": 0/3,
                                "att3=high": 1/3,
                                "att3=normal": 2/3,
                                "att4=none": 1/3,
                                "att4=slight": 0/3,
                                "att4=heavy": 2/3},
                             "cancelled": {
                                "att1=weekday": 0/1,
                                "att1=saturday": 1/1,
                                "att1=sunday": 0/1,
                                "att1=holiday": 0/1,
                                "att2=spring": 1/1,
                                "att2=summer": 0/1,
                                "att2=autumn": 0/1,
                                "att2=winter": 0/1,
                                "att3=none": 0/1,
                                "att3=high": 1/1,
                                "att3=normal": 0/1,
                                "att4=none": 0/1,
                                "att4=slight": 0/1,
                                "att4=heavy": 1/1}}
    nb_clf_bramer = MyNaiveBayesClassifier()
    nb_clf_bramer.fit(X_train_bramer, y_train_bramer)
    assert nb_clf_bramer.priors == priors_sol_bramer
    assert nb_clf_bramer.posteriors == posteriors_sol_bramer

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    X_train_inclass_example = [ # header = ["att1", "att2"]
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_inclass_example = [[1, 5]]
    y_test_inclass_example = ["yes"]
    nb_clf_inclass_ex = MyNaiveBayesClassifier()
    nb_clf_inclass_ex.fit(X_train_inclass_example, y_train_inclass_example)
    clf_predictions_inclass_ex = nb_clf_inclass_ex.predict(X_test_inclass_example)
    assert clf_predictions_inclass_ex == y_test_inclass_example

    # RQ5 (fake) iPhone purchases dataset
    X_train_iphone = [ # header = ["standing", "job_status", "credit_rating", "buys_iphone"]
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]
    X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_test_iphone = ["yes", "no"]
    nb_clf_iphone = MyNaiveBayesClassifier()
    nb_clf_iphone.fit(X_train_iphone, y_train_iphone)
    y_predicitons_iphone = nb_clf_iphone.predict(X_test_iphone)
    assert y_predicitons_iphone == y_test_iphone

    # Bramer 3.2 train dataset
    X_train_bramer = [ # header = ["day", "season", "wind", "rain", "class"]
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test_bramer = [["weekday", "winter", "high", "heavy"], ["weekday", "summer", "high", "heavy"], \
        ["sunday", "summer", "normal", "slight"]]
    y_test_bramer = ["very late", "on time", "on time"]
    nb_clf_bramer = MyNaiveBayesClassifier()
    nb_clf_bramer.fit(X_train_bramer, y_train_bramer)
    y_predictions_bramer = nb_clf_bramer.predict(X_test_bramer)
    assert y_predictions_bramer == y_test_bramer

def test_decision_tree_classifier_fit():
    X_train_interview = [ # header = ["level", "lang", "tweets", "phd", "interviewed_well"]
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]
    tree_interview = \
        ["Attribute", "att1",
            ["Value", "Junior",
                ["Attribute", "att4",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]
    dt_clf_interview = MyDecisionTreeClassifier()
    dt_clf_interview.fit(X_train_interview, y_train_interview)
    assert dt_clf_interview.tree == tree_interview

    X_train_iphone = [ # header = ["standing", "job_status", "credit_rating", "buys_iphone"]
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]
    tree_iphone = \
        ["Attribute", "att1",
            ["Value", "1",
                ["Attribute", "att2",
                    ["Value", "1",
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", "2",
                        ["Attribute", "att3",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", "3",
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value","2",
                ["Attribute", "att3",
                    ["Value", "excellent",
                        ["Leaf", "no", 4, 10]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]
    dt_clf_iphone = MyDecisionTreeClassifier()
    dt_clf_iphone.fit(X_train_iphone, y_train_iphone)
    assert dt_clf_iphone.tree == tree_iphone

def test_decision_tree_classifier_predict():
    X_train_interview = [ # header = ["level", "lang", "tweets", "phd", "interviewed_well"]
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]
    X_test_interview = [["Junior", "Java", "yes", "no"],
                        ["Junior", "Java", "yes", "yes"]]
    y_test_interview = ["True", "False"]
    dt_clf_interview = MyDecisionTreeClassifier()
    dt_clf_interview.fit(X_train_interview, y_train_interview)
    interview_predictions = dt_clf_interview.predict(X_test_interview)
    assert interview_predictions == y_test_interview

    X_train_iphone = [ # header = ["standing", "job_status", "credit_rating", "buys_iphone"]
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]
    X_test_iphone = [["1", "2", "excellent"],
                     ["2", "1", "fair"],
                     ["2", "2", "excellent"]]
    y_test_iphone = ["yes", "yes", "no"]
    dt_clf_iphone = MyDecisionTreeClassifier()
    dt_clf_iphone.fit(X_train_iphone, y_train_iphone)
    iphone_predictions = dt_clf_iphone.predict(X_test_iphone)
    assert iphone_predictions == y_test_iphone

def test_random_forest_classifier_fit():
    X_interview = [  # header = ["level", "lang", "tweets", "phd", "interviewed_well"]
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]]
    y_interview = ["False", "False", "True", "True", "True", "False", "True",
                   "False", "True", "True", "True", "True", "True", "False"]
    
    splits = myevaluation.stratified_split(
        X_interview, y_interview, n_splits=3)
    X_test = [X_interview[idx0] for idx0 in splits[0]]
    y_test = [y_interview[idx0] for idx0 in splits[0]]
    X_train = [X_interview[idx1] for idx1 in splits[1]] + \
        [X_interview[idx2] for idx2 in splits[2]]
    y_train = [y_interview[idx1] for idx1 in splits[1]] + \
        [y_interview[idx2] for idx2 in splits[2]]
    #print("X_test:", X_test, "\ny_test:", y_test, "\nX_train:", X_train, "\ny_train:", y_train)
    rf_clf = MyRandomForestClassifier(5, 2, 2)
    rf_clf.fit(X_train, y_train)
    rf_clf.predict(X_test)

    print("best trees:")
    for tree in rf_clf.trees:
        print(tree.tree)

    

def test_random_forest_classifier_predict():
    assert True == False