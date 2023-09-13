import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
import gradient_descent_helpers as gd
import plot_helpers as plth

#read files
dfTrain = pd.read_csv("data/StoreSales/train_2017.csv")
dfHolidays = pd.read_csv("data/StoreSales/holidays_events.csv")
dfOil = pd.read_csv("data/StoreSales/oil.csv")
df = pd.merge(dfTrain, dfHolidays, how="left", on="date")
df = pd.merge(df, dfOil, how="left", on="date")
#df=df.loc[df["date"].str.slice(0,4) == "2017"]
#print(df)


#columns
df = df[["id","date","store_nbr","family","sales","onpromotion","type","dcoilwtico"]]
#date
df["year"] = df["date"].str.slice(0,4).astype(int)
df["month"] = df["date"].str.slice(5, 7).str.lstrip("0").astype(int)
df["day"] = df["date"].str.slice(8, 10).str.lstrip("0").astype(int)
                     

#family
family = list(df["family"].drop_duplicates())
familyMap = {string: position for position, string in enumerate(family)}
df["family_num"] = df["family"].map(familyMap)
#holiday
df.loc[df['type'].isna()==False, 'is_holiday'] = 1
df["onpromotion"] = df["onpromotion"].astype(float)
#NaN
df.fillna(0, inplace=True)
#zscore
df_o = df[["year", "month", "day","store_nbr","family_num","onpromotion","dcoilwtico", "is_holiday"]]
df_z = df_o.apply(zscore)
#df_z = df[["year", "month", "day","store_nbr","family_num","onpromotion","dcoilwtico", "is_holiday"]].apply(zscore)
# print(df[["year", "month", "day","store_nbr","family_num","onpromotion","dcoilwtico", "is_holiday"]].min(0))
# print(df[["year", "month", "day","store_nbr","family_num","onpromotion","dcoilwtico", "is_holiday"]].max(0))
#print(df_z)

# #gradient descent
y_vect = df["sales"].to_numpy()
x_matrix = df_z.to_numpy()
w_init = np.zeros(x_matrix.shape[1])
b_init = 0.0

w_vector, b, cost_history = gd.run_linear_gd(x_matrix, y_vect, w_init, b_init, 100, 0.1)
print(w_vector, b, cost_history[-1])
plth.plot_vect(cost_history, equal_scale=False)