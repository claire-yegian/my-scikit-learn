##############################################
# Programmer: Claire Yegian
# Class: CPSC 322-01, Fall 2022
# Programming Assignment #7
# 11/21/22
# Description: basic functions for five classifiers: simple linear regressoion,
# k nearest neighbors, dummy, naive bayes, and decision tree
##############################################
import numpy as np
import os

from mysklearn import myutils

class MySimpleLinearRegressor:
    """Represents a simple linear regressor. Used in the MySimpleLinearRegressionClassifier
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in X_test:
                predictions.append(self.slope * test_instance[0] + self.intercept)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.
        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values
        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
            / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted_numeric = self.regressor.predict(X_test)
        y_predicted = []
        for pred in y_predicted_numeric:
            y_predicted.append(self.discretizer(pred))
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test, shuffle=False):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            shuffle(boolean): should the list of neighbors be sorted before picking k
                nearest neighbors? True if yes, False if no. Useful when classifying on
                few attributes because it prevents the classifier from always making the
                same prediction
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        if self.X_train is None or self.y_train is None:
            raise Exception("Make sure classifier is fit with X_train and y_train")
        # collect the distances for all neighbors for all instances of X_test
        all_distances = []
        for point in X_test:
            point_distances = {}
            for i in range(len(self.X_train)):
                point_distances[i] = myutils.compute_distance(point, self.X_train[i])
            all_distances.append(point_distances)
        # sort the distances by size
        all_dists_sorted = []
        for i in range(len(all_distances)):
            dists_sorted = list(all_distances[i].items())
            if shuffle:
                myutils.randomize_in_place(dists_sorted)
            dists_sorted = sorted(dists_sorted, key=lambda kv: kv[1])
            dists_sorted = dists_sorted[:self.n_neighbors]
            dists_sorted = sorted(dists_sorted, key=lambda y: y[0])
            all_dists_sorted.append(dists_sorted)
        # create lists of the distances and their respective indicies
        distances, indicies = [], []
        for dists in all_dists_sorted:
            distances.append([item[1] for item in dists])
            indicies.append([item[0] for item in dists])
        return distances, indicies

    def predict(self, X_test, shuffle=False):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            shuffle(boolean): should the list of neighbors be sorted before picking k
                nearest neighbors? True if yes, False if no. Useful when classifying on
                few attributes because it prevents the classifier from always making the
                same prediction
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.X_train is None or self.y_train is None:
            raise Exception("Make sure classifier is fit with X_train and y_train")
        _, indicies = self.kneighbors(X_test, shuffle)
        predictions = []
        for i in range(len(X_test)):
            neighbors = [self.y_train[index] for index in indicies[i]]
            most_common = max(set(neighbors), key=neighbors.count)
            predictions.append(most_common)
        return predictions

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = max(set(y_train), key=y_train.count)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            raise Exception("Model not fitted")

        return [self.most_common_label for instance in X_test]

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}
        for value in y_train:
            if value not in self.priors:
                self.priors[value] = 1 #count classifications
                self.posteriors[value] = {} #set up external posteriors dictionary
            else:
                self.priors[value] += 1
        for key in self.priors:
            self.priors[key] = self.priors[key]/len(y_train) #turn counts into priors

        attributes = myutils.get_attributes_dict(X_train[0], X_train)
        for class_key in self.posteriors:
            att_dict = {}
            for att_key in attributes:
                for occurance in attributes[att_key]:
                    att = att_key.split("tt")[1]
                    # att - 1 is the index of the attribute add "att_=_": probability pairs to
                    # attribute dictionary for each occurance of each attribute for each classification
                    att_dict[att_key + '=' + str(occurance)] = \
                        myutils.count_occurances(X_train, (int(att) - 1), occurance, \
                        y_train, class_key)/(self.priors[class_key]*len(y_train))
            self.posteriors[class_key] = att_dict

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            att_list = []
            for attribute in instance:
                att_list.append(attribute)
            #list of keys to retrieve
            att_list = ["att" + str(i+1) + "=" + str(att_list[i]) for i in range(len(att_list))]
            max_prob = 0
            prediction = ""
            for classification in self.priors:
                #list of posteriors for each attribute of the instance
                instance_posts = [self.posteriors[classification][att_list[i]] for i in range(len(att_list))]
                probability = np.product(instance_posts) * self.priors[classification]
                if probability > max_prob: #keep track of the highest probability classification
                    max_prob = probability
                    prediction = classification
            predictions.append(prediction)
        return predictions

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.attribute_domains = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))] # class label now at instance[-1]
        self.header = ["att" + str(i + 1) for i in range(len(train[0]))]
        self.attribute_domains = myutils.get_attributes_dict(self.header, train)
        available_attributes = self.header.copy()[:-1]
        self.tree = myutils.tdidt(train, available_attributes, self.header, self.attribute_domains, len(train))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            predictions.append(myutils.tdidt_predict(self.tree, instance, self.header))
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att1", "att2", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = myutils.tdidt_write_rules(self.tree, attribute_names, class_name, self.header)
        for rule in rules:
            rule_as_str = "IF "
            for i in range(len(rule) - 2):
                rule_as_str += rule[i] + " AND "
            rule_as_str += rule[-2] + " THEN " + rule[-1]
            print(rule_as_str)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        f = open(dot_fname, "w")
        f.write("graph g { \n")
        lines = myutils.make_dotfile(self.tree, [], 0, None, None, [])
        for line in lines:
            if type(line) == str:
                f.write(line)
        f.write("}")
        f.close()
        os.popen("dot -Tpdf -o " + pdf_fname + " " + dot_fname)