import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

#wrapper na funkciu, ktora zvacsi plot
def f(figsize):
    if figsize is not None:
        plt.figure(figsize=figsize)


########################## ANALYZA JEDNEHO ATRIBUTU ##########################

#vrati pocetnost jednotlivych kategorii a podla clz - countplot (nieco ako barplot)
def univariate_categ_to_class(data, categ_attr, clz="class", figsize=None):
    print(data[categ_attr].value_counts())

    f(figsize)
    sns.countplot(data=data, y=categ_attr, hue=clz)

#vlastnosti distribucie + boxplot + distplot
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

#vrati vztah medzi atributom a clz - korelacia + violinplot + distplot + regplot (vrati logisticku regresiu,
#pokial je tam dostatocne velka korelacia)
def univariate_num_to_class(data, num_attr, clz="class", threshold=0.4, figsize=None, show_regplot=True):
    corr = data[[num_attr, clz]].corr().iloc[0, 1]
    print("Korelacia medzi nasim atributom a y:", corr)

    f(figsize)

    sns.violinplot(data=data, y=num_attr, x=clz)
    plt.show()

    f(figsize)

    sns.distplot(data[data[clz] == 0][num_attr], bins=20, label=clz + "0")
    sns.distplot(data[data[clz] == 1][num_attr], bins=20, label=clz + "1")
    plt.legend()
    plt.show()

    if (corr > threshold or corr < -threshold) and show_regplot == True:
        f(figsize)
        sns.regplot(data=data, x=num_attr, y=clz, logistic=True)
        plt.show()

#vrati podiel daneho atributu podla clz
#konkretne pre tieto default values to vrati podiel reprezentujuci pocet cukrovkarou s jednotlivych
#hodnot daneho atributu attr
def get_percent_of_class_in_attr(data, attr, class_col="class", class_value=1):
    left = data[attr].value_counts().to_frame()
    right = data[data[class_col] == class_value][attr].value_counts().to_frame()

    lsuffix = "-total"
    rsuffix = "-" + class_col + str(class_value)

    leftname = attr + lsuffix
    rightname = attr + rsuffix

    joined = left.join(right, how="left", lsuffix=lsuffix, rsuffix=rsuffix).fillna(0)

    return (joined[rightname] / joined[leftname]).sort_values(ascending=False)


########################## PAROVA ANALYZA ##########################

#medzi 2 ciselnymi atributmi - korelacia + scatterplot + regplot (if necessary)
def bivar_2nums(data, x, y, hue="class", figsize=None, show_regplot=True):
    clean = data.loc[(data[x].notna()) & (data[y].notna())]

    cor = stats.pearsonr(clean[x], clean[y])[0]
    print("Pearsonova korelacia:", cor)

    f(figsize)
    sns.scatterplot(data=data, x=x, y=y, hue="class")
    plt.show()

    if (cor >= 0.7 or cor <= -0.7) and show_regplot == True:
        regress = stats.linregress(clean[x], clean[y])
        print(regress)

        f(figsize)
        sns.regplot(data[x], data[y])
        plt.show()


#medzi 2 kategorickymi atributmi - vytvori sa pocetnostna tabulka, ktora sluzi na vytvorenie heatmapy
#mozeme taktiez zobrazit countplot - vhodne, ked je malo, 2-3 hodnoty z jedneho atributu
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


#medzi jednym ciselnym a jednym kategorickym atributom - boxplot + violinplot + distplot
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
        
        
def heatmap_mask(data, threshold=0.4):
    data = data.copy()
    
    for i in data.index:
        clean = False
        row = data.loc[i]
        for c in data.columns:
            if i == c:
                clean = True
                
            if clean == True:
                row[c] = 1
                
            else:
                val = row[c]
                if val == 1 or val < threshold and val > -threshold:
                    row[c] = 1
                else:
                    row[c] = 0

    data = data.astype("bool")
    return data