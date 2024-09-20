import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import streamlit as st

df = pd.read_csv("DataFrame.csv", nrows=1000)

st.title("Mon application de Prédiction F1")
st.write("Bonjour, bienvenue dans votre application de prédiction du classement des pilotes !")

df_prediction = pd.read_csv('DataFrame.csv')  
weather_df = pd.read_parquet('weather.parquet')

weather_df['apply_time_rl'] = pd.to_datetime(weather_df['apply_time_rl'], unit='s')
df_prediction['date'] = pd.to_datetime(df_prediction['date'], errors='coerce')

df_prediction = df_prediction.merge(
    weather_df,
    left_on='date',
    right_on='apply_time_rl',
    how='left'
)

X = df_prediction[['grid', 'fastestLapTime', 'points', 'q1', 'circuitId', 'driverId']].copy()

X['q1'] = pd.to_numeric(X['q1'].str.replace(':', ''), errors='coerce')
X['fastestLapTime'] = pd.to_numeric(X['fastestLapTime'].str.replace(':', ''), errors='coerce')

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

y = df_prediction['positionOrder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R² Score: {r2}")

df_results = pd.DataFrame({'Réel': y_test, 'Prédit': y_pred})

st.write("Classement réel et prédit des pilotes :")
st.write(df_results.head())  
st.write("Comparaison des valeurs réelles et prédites :")
plt.figure(figsize=(10,6))
plt.plot(df_results['Réel'].values, label='Réel', color='blue')
plt.plot(df_results['Prédit'].values, label='Prédit', color='red')
plt.title('Comparaison des valeurs réelles et prédites')
plt.legend()
st.pyplot(plt)

plt.figure(figsize=(10,6))
sns.histplot(df_results['Erreur'], bins=20, kde=True)
plt.title('Distribution des erreurs (Réel - Prédit)')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
st.pyplot(plt)

plt.figure(figsize=(10,6))
sns.scatterplot(x=df_results['Réel'], y=df_results['Prédit'])
plt.plot([df_results['Réel'].min(), df_results['Réel'].max()], [df_results['Réel'].min(), df_results['Réel'].max()], 'k--', lw=2)
plt.title('Valeurs réelles vs Valeurs prédites')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
st.pyplot(plt)

dump(model, 'race_position_predictor_model.joblib')
dump(imputer, 'imputer.joblib')

st.write("Le modèle a été entraîné et sauvegardé avec succès pour prédire le classement des pilotes.")

