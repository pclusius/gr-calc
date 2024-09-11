
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#median filter window 5
path_der_filterafter = r"./1st_derivatives_filterafter.csv"
path_der_filterbefore = r"./1st_derivatives_filterbefore.csv"
path_der_nofilter = r"./1st_derivatives_nofilter.csv"
path_der_logconc_filterafter = r"./1st_derivatives_logconc_filterafter.csv"
path_der_logconc_filterbefore = r"./1st_derivatives_logconc_filterbefore.csv"
path_der_logconc_nofilter = r"./1st_derivatives_logconc_nofilter.csv"
path_nofilter = r"./data_format_filter.csv"
path_filter = r"./data_format_nofilter.csv"

data_der_filterafter = pd.read_csv(path_der_filterafter, sep=',')
data_der_filterbefore = pd.read_csv(path_der_filterbefore, sep=',')
data_der_nofilter = pd.read_csv(path_der_nofilter, sep=',')
data_der_logconc_filterafter = pd.read_csv(path_der_logconc_filterafter, sep=',')
data_der_logconc_filterbefore = pd.read_csv(path_der_logconc_filterbefore, sep=',')
data_der_logconc_nofilter = pd.read_csv(path_der_logconc_nofilter, sep=',')
data_nofilter = pd.read_csv(path_nofilter, sep=',')
data_filter = pd.read_csv(path_filter, sep=',')

df1 = pd.DataFrame(data_der_filterafter)
df2 = pd.DataFrame(data_der_filterbefore)
df3 = pd.DataFrame(data_der_nofilter)
df4 = pd.DataFrame(data_der_logconc_filterafter)
df5 = pd.DataFrame(data_der_logconc_filterbefore)
df6 = pd.DataFrame(data_der_logconc_nofilter)
df7 = pd.DataFrame(data_nofilter)
df8 = pd.DataFrame(data_filter)

df1['Timestamp (UTC)']=pd.to_datetime(df1['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df2['Timestamp (UTC)']=pd.to_datetime(df2['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df3['Timestamp (UTC)']=pd.to_datetime(df3['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df4['Timestamp (UTC)']=pd.to_datetime(df4['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df5['Timestamp (UTC)']=pd.to_datetime(df5['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df6['Timestamp (UTC)']=pd.to_datetime(df6['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df7['Timestamp (UTC)']=pd.to_datetime(df7['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")
df8['Timestamp (UTC)']=pd.to_datetime(df8['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S.%f")

#plot
fig, ax = plt.subplots(3,figsize=(14, 15), dpi=90)

#9.0794158
#7.5459950000000005
#30.834151000000002

x1 = df1.iloc[:,0]
y1 = df1["30.834151000000002"]
ax[1].plot(x1,y1,linestyle='dotted', color="black")

x2 = df2.iloc[:,0]
y2 = df2["30.834151000000002"]
ax[1].plot(x2,y2,linestyle='-')

x3 = df3.iloc[:,0]
y3 = df3["30.834151000000002"]
ax[1].plot(x3,y3,linestyle='--')

x4 = df4.iloc[:,0]
y4 = df4["30.834151000000002"]
ax[2].plot(x4,y4,linestyle='dotted', color="black")

x5 = df5.iloc[:,0]
y5 = df5["30.834151000000002"]
ax[2].plot(x5,y5,linestyle='-')

x6 = df6.iloc[:,0]
y6 = df6["30.834151000000002"]
ax[2].plot(x6,y6,linestyle='--')

x7 = df7.iloc[:,0]
y7 = df7["30.834151000000002"]
ax[0].plot(x7,y7,linestyle='-')

x8 = df8.iloc[:,0]
y8 = df8["30.834151000000002"]
ax[0].plot(x8,y8,linestyle='--')

ax[1].axhline(y=0.03, color='red', linestyle='-', linewidth=1)
ax[1].axhline(y=-0.03, color='red', linestyle='-', linewidth=1)
ax[2].axhline(y=0.000009, color='red', linestyle='-', linewidth=1)
ax[2].axhline(y=-0.000009, color='red', linestyle='-', linewidth=1)

ax[2].set_xlabel("Time (UTC)")
ax[0].set_ylabel("dN/dlogDp")
ax[1].set_ylabel("d(dN/dlogDp)dt")
ax[2].set_ylabel("dlog(dN/dlogDp)dt")
ax[0].legend(["no filter","filtered"])
ax[1].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.03"])
ax[2].legend(["deriv(filter)","deriv(no filter)","filter(deriv(no filter))","threshold=0.000009"])

ax[0].set_title("Dp = 30.834151000000002nm, median filter 5")
#plt.tight_layout()
plt.show()