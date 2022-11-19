import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df=pd.read_csv("D:\\T1.csv")
df.rename(columns={"Date/Time":"Time",
                            "LV ActivePower (kW)":"ActivePower(kw)",
                            "Wind Speed (m/s)":"WindSpeed(m/s)",
                            "Wind Direction (°)" :"Wind_Direction"},
                           inplace=True)
sns.pairplot(df)
corr=df.corr()
plt.figure(figsize=(10,8))
ax=sns.heatmap(corr,vmin=-1,vmax=1,annot=True)
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.show()
corr
df["Time"]=pd.to_datetime(df["Time"],format="%d %m %Y %H:%M",errors="coerce")
df
y=df["ActivePower(kw)"]
x=df[["Theoretical_Power_Curve (KWh)","WindSpeed(m/s)"]]
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x,y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
forest_model = RandomForestRegressor(max_leaf_nodes=500, random_state=1) 
forest_model.fit(train_x, train_y)
power_preds = forest_model.predict(val_x)
print(mean_absolute_error(val_y, power_preds))
print(r2_score(val_y,power_preds))
joblib.dump(forest_model,"power_prediction.sav")