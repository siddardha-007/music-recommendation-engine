from difflib import get_close_matches

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Music Recommendation Engine",
    page_icon="🎵",
    layout="wide"
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

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


# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/songs_processed.csv"
    )

    # Remove songs without title or artist
    df = df.dropna(
        subset=["track_name", "artist_name"]
    ).reset_index(drop=True)

    # Create cleaned title column
    df["clean_title"] = (
        df["track_name"]
        .str.strip()
        .str.lower()
    )

    return df


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    model = joblib.load(
        "models/recommender.pkl"
    )

    return scaler, model


# --------------------------------------------------
# Load everything
# --------------------------------------------------

df = load_data()

scaler, model = load_model()

# Scale features only once
X_scaled = scaler.transform(
    df[FEATURES]
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎵 Music Recommendation Engine")

st.write(
    "Discover songs based on their acoustic characteristics."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

number_of_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# --------------------------------------------------
# Song Search
# --------------------------------------------------

selected_song = st.text_input(
    "Enter a song name",
    placeholder="e.g. Blinding Lights"
)


# --------------------------------------------------
# Recommendation
# --------------------------------------------------

if st.button("🎧 Recommend Songs"):

    if not selected_song.strip():

        st.warning(
            "Please enter a song name."
        )

    else:

        query = selected_song.strip().lower()

        # ------------------------------------------
        # Find exact match
        # ------------------------------------------

        matches = df[
            df["clean_title"] == query
        ]

        # ------------------------------------------
        # Fuzzy match if exact match doesn't exist
        # ------------------------------------------

        if matches.empty:

            titles = df["clean_title"].tolist()

            close_matches = get_close_matches(
                query,
                titles,
                n=1,
                cutoff=0.5
            )

            if close_matches:

                matched_title = close_matches[0]

                matches = df[
                    df["clean_title"] == matched_title
                ]

                st.info(
                    f"🔍 Showing results for "
                    f"**{matches.iloc[0]['track_name']}**"
                )

        # ------------------------------------------
        # Song not found
        # ------------------------------------------

        if matches.empty:

            st.error(
                f"❌ '{selected_song}' was not found."
            )

        else:

            # Use first matching song
            song_index = matches.index[0]

            song = df.iloc[song_index]

            # --------------------------------------
            # Find similar songs
            # --------------------------------------

            target_vector = X_scaled[
                song_index
            ].reshape(1, -1)

            distances, indices = model.kneighbors(
                target_vector,
                n_neighbors=number_of_recommendations + 1
            )

            # --------------------------------------
            # Display title
            # --------------------------------------

            st.subheader(
                f"Recommendations similar to "
                f"'{song['track_name']}'"
            )

            # --------------------------------------
            # Build recommendations
            # --------------------------------------

            recommendations = []

            for distance, index in zip(
                distances[0][1:],
                indices[0][1:]
            ):

                recommended_song = df.iloc[index]

                similarity = (
                    1 - distance
                ) * 100

                recommendations.append({
                    "Song": recommended_song["track_name"],
                    "Artist": recommended_song["artist_name"],
                    "Similarity (%)": round(
                        similarity,
                        2
                    )
                })

            # --------------------------------------
            # Display results
            # --------------------------------------

            results_df = pd.DataFrame(
                recommendations
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )