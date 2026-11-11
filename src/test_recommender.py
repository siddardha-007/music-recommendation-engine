import pandas as pd
import joblib


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


df = pd.read_csv(
    "data/processed/songs_processed.csv"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

model = joblib.load(
    "models/recommender.pkl"
)


X = scaler.transform(
    df[FEATURES]
)


song_name = "Blinding Lights"

matches = df[
    df["track_name"].str.lower()
    == song_name.lower()
]


if matches.empty:

    print("Song not found.")

else:

    index = matches.index[0]

    distances, indices = model.kneighbors(
        X[index].reshape(1, -1),
        n_neighbors=11
    )

    print("\nRecommendations:\n")

    for distance, idx in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        similarity = 1 - distance

        print(
            df.iloc[idx]["track_name"],
            "-",
            df.iloc[idx]["artist_name"],
            f"({similarity:.2%})"
        )