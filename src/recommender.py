import numpy as np
from sklearn.neighbors import NearestNeighbors

class MusicRecommender:
    def __init__(self, data):
        self.model = NearestNeighbors(
            n_neighbors=5,
            algorithm='brute', 
            metric='cosine'
        )

        self.features = None
        self.songs = None

    def fit(self, X, songs):

        self.features = X
        self.songs = songs.reset_index(drop=True)

        self.model.fit(X)

        return self

    def recommend(self, song_index, n_recommendations=10):

        distances, indices = self.model.kneighbors(
            self.feaures[song_index].reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )

        recommendations = []

        for distance, index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            similarity = 1 - distance

            song = self.songs.iloc[index].copy()

            song["similarity"] = similarity

            recommendations.append(song)

        return recommendations