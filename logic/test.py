import pandas as pd
import numpy as np

# CSV einlesen
df = pd.read_csv("data/FNG.csv")

# timestamp_nano in Sekunden umrechnen und zu datetime konvertieren
df["timestamp"] = pd.to_datetime(df["timestamp_nano"], unit="ns")

# Format anpassen (YYYY-MM-DD HH:MM:SS)
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Alte Spalten löschen
df = df.drop(columns=["timestamp_nano", "timestamp_iso"])

# Neue Spalten mit Zufallswerten hinzufügen
np.random.seed(42)  # für reproduzierbare Werte, falls gewünscht
df["rsi"] = np.random.uniform(0, 100, size=len(df)).round(2)
df["alt_coin_season_index"] = np.random.uniform(0, 1, size=len(df)).round(3)

# Ergebnis speichern
df.to_csv("fear_greed_clean.csv", index=False)

print(df.head())
