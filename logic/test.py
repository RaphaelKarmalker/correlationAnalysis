import pandas as pd
import numpy as np

def add_dummy_volume(input_path: str, output_path: str = "with_volume.csv"):
    # CSV einlesen
    df = pd.read_csv(input_path)

    # Neue Spalte 'volume' mit Zufallswerten hinzufügen (z. B. 1000–10000)
    np.random.seed(42)  # für Reproduzierbarkeit
    df["volume"] = np.random.randint(1000, 10000, size=len(df))

    # Ergebnis speichern
    df.to_csv(output_path, index=False)
    print(f"Datei gespeichert unter: {output_path}")
    print(df.head())

# Beispielaufruf
add_dummy_volume("data/bars-5M.csv", "data/ohlc_with_volume.csv")
