import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import streamlit as st

# Chargement des données
df = pd.read_csv("DataFrame.csv", nrows=1000)

st.title("Mon application de Prédiction F1")
st.write("Bonjour, bienvenue dans votre application de prédiction du classement des pilotes !")

df_prediction = pd.read_csv('DataFrame.csv')  
weather_df = pd.read_parquet('weather.parquet')

# Conversion des dates
weather_df['apply_time_rl'] = pd.to_datetime(weather_df['apply_time_rl'], unit='s')
df_prediction['date'] = pd.to_datetime(df_prediction['date'], errors='coerce')

# Fusion des données
df_prediction = df_prediction.merge(
    weather_df,
    left_on='date',
    right_on='apply_time_rl',
    how='left'
)

# Préparation des features
X = df_prediction[['grid', 'fastestLapTime', 'points', 'q1', 'circuitId', 'driverId']].copy()

# Conversion des colonnes en valeurs numériques
X['q1'] = pd.to_numeric(X['q1'].str.replace(':', ''), errors='coerce')