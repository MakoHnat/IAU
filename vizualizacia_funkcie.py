import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def f(figsize):
    if figsize is not None:
        plt.figure(figsize=figsize)


def univariate_categ_to_class(data, categ_attr, cls="class", figsize=None):
    print(data[categ_attr].value_counts())

    f(figsize)
    sns.countplot(data=data, y=categ_attr, hue=cls)


def univariate_num(data, num_attr, figsize=None):
    print(data[num_attr].describe())

    print("Skewness:", stats.skew(data[num_attr], nan_policy="omit").data)
    print("Kurtosis:", stats.kurtosis(data[num_attr], nan_policy="omit"))

    mean = data[num_attr].mean()
    median = data[num_attr].median()
    mode = data[num_attr].mode()[0]

    f(figsize)

    sns.boxplot(data=data, x=num_attr)
    plt.show()

    f(figsize)

    plt.axvline(mean, linestyle='-', label="mean")
    plt.axvline(median, linestyle='--', label="median")
    plt.axvline(mode, linestyle=':', label="mode")

    sns.distplot(data[num_attr], bins=20)
    plt.legend()
    plt.show()


def univariate_num_to_class(data, num_attr, cls="class", threshold=0.4, figsize=None):
    corr = data[[num_attr, cls]].corr().iloc[0, 1]
    print("Korelacia medzi nasim atributom a y:", corr)

    f(figsize)

    sns.violinplot(data=data, y=num_attr, x=cls)
    plt.show()

    f(figsize)

    sns.distplot(data[data[cls] == 0][num_attr], bins=20, label=cls + "0")
    sns.distplot(data[data[cls] == 1][num_attr], bins=20, label=cls + "1")
    plt.legend()
    plt.show()

    if corr > threshold or corr < -threshold:
        f(figsize)
        sns.regplot(data=data, x=num_attr, y=cls, logistic=True)
        plt.show()


def get_percent_of_class_in_attr(data, attr, class_col="class", class_value=1):
    left = data[attr].value_counts().to_frame()
    right = data[data[class_col] == class_value][attr].value_counts().to_frame()

    lsuffix = "-total"
    rsuffix = "-" + class_col + str(class_value)

    leftname = attr + lsuffix
    rightname = attr + rsuffix

    joined = left.join(right, how="left", lsuffix=lsuffix, rsuffix=rsuffix).fillna(0)

    return (joined[rightname] / joined[leftname]).sort_values(ascending=False)


def bivar_2nums(data, x, y, hue="class", figsize=None):
    clean = data.loc[(data[x].notna()) & (data[y].notna())]

    cor = stats.pearsonr(clean[x], clean[y])[0]
    print("Pearsonova korelacia:", cor)

    f(figsize)
    sns.scatterplot(data=data, x=x, y=y, hue="class")
    plt.show()

    if cor >= 0.7 or cor <= -0.7:
        regress = stats.linregress(clean[x], clean[y])
        print(regress)

        f(figsize)
        sns.regplot(data[x], data[y])
        plt.show()


def bivar_2cats(data, x, y, show_hue=False, hue="class", countplot=False, figsize=None):
    if show_hue == False:

        table = data.groupby(by=[y])[x].value_counts()
        table = table.unstack().fillna(0).astype("int32")

    else:

        table = data.groupby(by=[y, x])[hue].value_counts()
        table = table.unstack().unstack().fillna(0).astype("int32")

    f(figsize)
    sns.heatmap(data=table, cmap="Blues", annot=True, fmt="d")
    plt.show()

    if countplot == True:
        f(figsize)
        sns.countplot(data=data, x=x, hue=y)
        plt.show()


def bivar_numcat(data, num, cat, hue="class", distplot=False, figsize=None):
    f(figsize)
    sns.boxplot(data=data, x=cat, y=num)
    plt.show()

    f(figsize)
    sns.violinplot(data=data, x=cat, y=num, hue=hue)
    plt.show()

    if distplot == True:

        f(figsize)

        for c in data[cat].unique():
            if c is np.nan:
                continue

            sns.distplot(data[data[cat] == c][num], label=cat + ":" + c)

        plt.legend()
        plt.show()