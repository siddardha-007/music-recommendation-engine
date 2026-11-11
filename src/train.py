import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


DATA_PATH = "data/raw/spotify_tracks.csv"

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

df = pd.read_csv(DATA_PATH)

if "track_id" in df.columns:

    df = df.drop_duplicates(
        subset=["track_id"]
    )

else:

    df = df.drop_duplicates(
        subset=["track_name", "artists"]
    )

df = df.dropna(
    subset=FEATURES
).reset_index(drop=True)



X = df[FEATURES]


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(X_scaled)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    model,
    "models/recommender.pkl"
)


df.to_csv(
    "data/processed/songs_processed.csv",
    index=False
)

print("Training completed.")
print("Songs:", len(df))