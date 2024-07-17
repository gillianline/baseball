#Gillian Line-Luttrell

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

df = pd.read_csv('battedBallData.csv')

df['average_speed'] = (df['speed_A'] + df['speed_B']) / 2

vangle_by_hittype = df.groupby('hittype')[['vangle_A', 'vangle_B']].mean()
vangle_by_hittype['average_vangle'] = (vangle_by_hittype['vangle_A'] + vangle_by_hittype['vangle_B']) / 2

df = pd.merge(df, vangle_by_hittype['average_vangle'], on='hittype', how='left')

df.loc[(df['hittype'] == 'fly_ball') & (df['average_vangle'] > 10), 'average_speed'] *= 1.25
df.loc[(df['hittype'] == 'fly_ball') & (df['average_vangle'] <= 10), 'average_speed'] *= 1.15
df.loc[df['hittype'] == 'ground_ball', 'average_speed'] *= 1.1

df['true_swing_speed'] = df['average_speed'] / 1.5

df.drop('average_vangle', axis=1, inplace=True)

average_true_swing_speed = df.groupby(['batter', 'hittype'])['true_swing_speed'].mean().reset_index()

next_season = pd.merge(df, average_true_swing_speed, on=['batter', 'hittype'], suffixes=('', '_next_season'))

imputer = SimpleImputer(strategy='mean')
X = next_season[['true_swing_speed']]
X_imputed = imputer.fit_transform(X)

y = next_season['true_swing_speed_next_season']
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1))

model = LinearRegression()
model.fit(X_imputed, y_imputed)

next_season['predicted'] = model.predict(X_imputed)

next_season['difference'] = next_season['predicted'] - next_season['true_swing_speed']


result_df = next_season.groupby(['batter', 'hittype'])[['true_swing_speed', 'predicted', 'difference']].mean().reset_index()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result_df)
