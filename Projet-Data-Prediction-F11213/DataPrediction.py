import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import streamlit as st
import dask.dataframe as dd

# Charger les données en utilisant Dask
weather_df = dd.read_parquet('weather.parquet')

weather_df = weather_df.compute()

# Chargement des autres données
df = pd.read_csv("DataFrame.csv", nrows=1000)

st.title("Mon application de Prédiction F1")
st.write("Bonjour, bienvenue dans votre application de prédiction du classement des pilotes !")

# Conversion des dates
weather_df['apply_time_rl'] = pd.to_datetime(weather_df['apply_time_rl'], unit='s')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Fusion des données
df = df.merge(
    weather_df,
    left_on='date',
    right_on='apply_time_rl',
    how='left'
)

# Préparation des features
X = df[['grid', 'fastestLapTime', 'points', 'q1', 'circuitId', 'driverId']].copy()

# Conversion des colonnes en valeurs numériques
X['q1'] = pd.to_numeric(X['q1'].str.replace(':', '', regex=False), errors='coerce')
X['fastestLapTime'] = pd.to_numeric(X['fastestLapTime'].str.replace(':', '', regex=False), errors='coerce')

# Gestion des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Cible (position)
y = df['positionOrder']

# Division des données en ensemble d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Calcul des métriques de performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R² Score: {r2}")

# Création d'un DataFrame avec les résultats réels et prédits
df_results = pd.DataFrame({'Réel': y_test, 'Prédit': y_pred})

# Calcul de l'erreur
df_results['Erreur'] = df_results['Réel'] - df_results['Prédit']

# Affichage du classement réel et prédit
st.write("Classement réel et prédit des pilotes :")
st.write(df_results.head())

# Comparaison des valeurs réelles et prédites
sns.lineplot(x=range(len(df_results)), y=df_results['Réel'], label='Réel', color='blue')
sns.lineplot(x=range(len(df_results)), y=df_results['Prédit'], label='Prédit', color='red')
plt.title('Comparaison des valeurs réelles et prédites')
st.pyplot()

# Distribution des erreurs
sns.histplot(df_results['Erreur'], bins=20, kde=True)
plt.title('Distribution des erreurs (Réel - Prédit)')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
st.pyplot()

# Visualisation des valeurs réelles vs prédites
sns.scatterplot(x=df_results['Réel'], y=df_results['Prédit'])
plt.plot([df_results['Réel'].min(), df_results['Réel'].max()], [df_results['Réel'].min(), df_results['Réel'].max()], 'k--', lw=2)
plt.title('Valeurs réelles vs Valeurs prédites')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
st.pyplot()

# Sauvegarde du modèle et de l'imputer
dump(model, 'race_position_predictor_model.joblib')
dump(imputer, 'imputer.joblib')

