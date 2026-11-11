# 🎵 Song & Playlist Recommendation Engine

A content-based music recommendation system that recommends songs based on their **audio characteristics** such as danceability, energy, loudness, acousticness, valence, tempo, and other Spotify-style audio features.

The project uses **Machine Learning with Nearest Neighbors and Cosine Similarity** to find songs that are acoustically similar to a selected song.

---

## 📌 Project Overview

Music streaming platforms such as Spotify, Apple Music, and YouTube Music need recommendation systems to help users discover songs they may enjoy.

This project builds a simple **content-based recommendation engine**.

Instead of recommending songs based on user ratings or listening history, the system compares songs using their audio features.

### Example

If the user enters:

> Blinding Lights

The system analyzes its audio characteristics and recommends songs with similar characteristics.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a real-world music recommendation system.
- Perform data cleaning and preprocessing.
- Analyze music/audio features using EDA.
- Select relevant features for recommendation.
- Normalize numerical features using `StandardScaler`.
- Build a content-based recommendation model.
- Use cosine distance to measure song similarity.
- Create a simple Streamlit web application.
- Provide similarity scores for recommended songs.

---

## 🧠 Recommendation Approach

This project uses a **Content-Based Recommendation System**.

The system does not depend on:

- User ratings
- User profiles
- Listening history
- Collaborative filtering

Instead, it recommends songs based on their audio features.

### Workflow

```text
Spotify Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Nearest Neighbors Model
       ↓
Cosine Similarity
       ↓
Song Recommendations
       ↓
Streamlit Dashboard