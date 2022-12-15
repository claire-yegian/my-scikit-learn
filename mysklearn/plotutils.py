"""
# Programmers: Claire Yegian and Anna Cardinal
# Class: CPSC 322-01, Fall 2022
# Final Project
# 12/14/22
# Description: helper functions for plotting data (used in our EDA)
"""

import copy
import matplotlib.pyplot as plt

from mysklearn import myutils

def simple_bar_chart(x_axis_data, y_axis_data, labels=None, bar_width=20, bar_height=5):
    """Makes a basic bar chart.

        Args:
            x_axis_data(list): list of x axis values
            y_axis_data(list of ints): parallel list of corresponding y values
            labels(list of strings): contains labels for x axis, y axis, and title, in that order
            bar_width(int): width of data bars
            bar_height(int): height of data bars
    """
    x_copy = copy.deepcopy(x_axis_data)
    y_copy = copy.deepcopy(y_axis_data)
    num_values = len(x_copy)
    for index in range(num_values):
        x_copy[index] = str(x_copy[index])
        y_copy[index] = int(y_copy[index])
    plt.figure(figsize=(bar_width, bar_height))
    if labels is None:
        plt.title("Title")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
    plt.title(labels[2])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.bar(x_copy, y_copy)
    plt.show()

def histogram(table, attribute, label):
    """Creates a histogram for an attribute of a MyPyTable object
    Args:
        table(MyPyTable): the MyPyTable object with the data
        attribute(str): the attribute we're finding and graphing the
            distribution of
        label(str): the label to use for the x axis and title
    """
    plt.figure(figsize = (8, 5.5))
    col = table.get_column(attribute)

    plt.hist(col, bins=10, edgecolor="white")
    plt.ylabel("Count")
    plt.xlabel(label)
    plt.title("Distribution of "+label)
    plt.show()

def scatter_plot(table, x, y, labels=None):
    """Creates a scatter plot for two attributes of a MyPyTable object
    Args:
        table(MyPyTable): the MyPyTable object with the data
        x(str or list): the attribute we're graphing on the x axis
        y(str or list): the attribute we're graphing on the y axis
        labels(list of str): if x and y are lists, labels is the string labels for the
            x and y axes of the graph
    Note: x and y must be of the same type: either both strings or both lists
    """
    plt.figure(figsize = (8, 5.5))
    if isinstance(x, str):
        labels = [x, y]
        x, y = table.get_column(x), table.get_column(y)
    plt.scatter(x, y)

    # Calculate m, b, correlation coefficient, and covariance (and plot the regression line)
    m, b = myutils.compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="red", lw=5)
    cor_coef = myutils.correlation_coefficient(x, y)
    cov = myutils.covariance(x, y)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[0]+" vs "+labels[1]+" (correlation coefficient: "+str(round(cor_coef, 4))+\
        " covariance: "+str(round(cov, 4))+")")
    plt.tight_layout()
    plt.show()
