
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def filtering_derivatives():
    #median filter window 5
    paths = [r"./1st_derivatives_filterafter.csv",r"./1st_derivatives_filterbefore.csv",r"./1st_derivatives_nofilter.csv",r"./1st_derivatives_logconc_filterafter.csv",r"./1st_derivatives_logconc_filterbefore.csv",r"./1st_derivatives_logconc_nofilter.csv",r"./data_format_filter.csv",r"./data_format_nofilter.csv"]

    dfs = []
    for path in paths:
        df = pd.DataFrame(pd.read_csv(path, sep=','))
        dfs.append(df)

    for df in dfs:
        df['Timestamp (UTC)']=pd.to_datetime(df['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")


    #plotting
    fig, ax = plt.subplots(3,figsize=(14, 15), dpi=90)

    #9.0794158
    #7.5459950000000005
    #30.834151000000002

    diameter ="30.834151000000002"

    y_list = []
    for i in dfs:
        y = df[diameter]
        y_list = y_list.append(y)

    x1 = df[0].iloc[:,0]
    y1 = y_list[0]
    ax[1].plot(x1,y1,linestyle='dotted', color="black")

    x2 = df[1].iloc[:,0]
    y2 = y_list[1]
    ax[1].plot(x2,y2,linestyle='-')

    x3 = df[2].iloc[:,0]
    y3 = y_list[2]
    ax[1].plot(x3,y3,linestyle='--')

    x4 = df[3].iloc[:,0]
    y4 = y_list[3]
    ax[2].plot(x4,y4,linestyle='dotted', color="black")

    x5 = df[4].iloc[:,0]
    y5 = y_list[4]
    ax[2].plot(x5,y5,linestyle='-')

    x6 = df[5].iloc[:,0]
    y6 = y_list[5]
    ax[2].plot(x6,y6,linestyle='--')

    x7 = df[6].iloc[:,0]
    y7 = y_list[6]
    ax[0].plot(x7,y7,linestyle='-')

    x8 = df[7].iloc[:,0]
    y8 = y_list[7]
    ax[0].plot(x8,y8,linestyle='--')

    ax[1].axhline(y=0.03, color='red', linestyle='-', linewidth=1)
    ax[1].axhline(y=-0.03, color='red', linestyle='-', linewidth=1)
    ax[2].axhline(y=0.000009, color='red', linestyle='-', linewidth=1)
    ax[2].axhline(y=-0.000009, color='red', linestyle='-', linewidth=1)

    ax[2].set_xlabel("Time (UTC)")
    ax.set_ylabel(["dN/dlogDp","d(dN/dlogDp)dt","dlog(dN/dlogDp)dt"])

    ax[0].legend(["no filter","filtered"])
    ax[1].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.03"])
    ax[2].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.000009"])

    ax[0].set_title(f"Dp = {diameter}nm, median filter 5")
    plt.show()


def appearance_time():
    #median filter window 5
    paths = [r"./data_format_nofilter.csv"]

    dfs = []
    for path in paths:
        df = pd.DataFrame(pd.read_csv(path, sep=','))
        dfs.append(df)

    for df in dfs:
        df['Timestamp (UTC)']=pd.to_datetime(df['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")


    #plotting
    fig, ax = plt.subplots(3,figsize=(14, 15), dpi=90)

    diameters = ["9.0794158","10.924603","26.678849999999997"]

    y_list = []
    for diam in diameters:
        y = df[diam]
        y_list = y_list.append(y)


    color = 'tab:red'
    
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

      # otherwise the right y-label is slightly clipped


    x1 = df[0].iloc[:,0]
    y1 = y_list[0]
    ax[1].plot(x1,y1,linestyle='dotted', color="black")

    x2 = df[1].iloc[:,0]
    y2 = y_list[1]
    ax[1].plot(x2,y2,linestyle='-')

    x3 = df[2].iloc[:,0]
    y3 = y_list[2]
    ax[1].plot(x3,y3,linestyle='--')

    x4 = df[3].iloc[:,0]
    y4 = y_list[3]
    ax[2].plot(x4,y4,linestyle='dotted', color="black")

    x5 = df[4].iloc[:,0]
    y5 = y_list[4]
    ax[2].plot(x5,y5,linestyle='-')

    x6 = df[5].iloc[:,0]
    y6 = y_list[5]
    ax[2].plot(x6,y6,linestyle='--')

    x7 = df[6].iloc[:,0]
    y7 = y_list[6]
    ax[0].plot(x7,y7,linestyle='-')

    x8 = df[7].iloc[:,0]
    y8 = y_list[7]
    ax[0].plot(x8,y8,linestyle='--')

    ax[1].axhline(y=0.03, color='red', linestyle='-', linewidth=1)
    ax[1].axhline(y=-0.03, color='red', linestyle='-', linewidth=1)

    ax[2].set_xlabel("Time (UTC)")
    ax.set_ylabel("dN/dlogDp", color="red")
    ax2.set_ylabel("log10(dN/dlogDp)", color="blue")

    ax[0].legend(["","filtered"])
    ax[1].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.03"])
    ax[2].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.000009"])

    ax[0].set_title(f"Dp = {diameter}nm, median filter 5")
    fig.tight_layout()
    plt.show()

appearance_time()



