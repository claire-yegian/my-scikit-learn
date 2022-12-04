import copy
import matplotlib.pyplot as plt
import numpy as np
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