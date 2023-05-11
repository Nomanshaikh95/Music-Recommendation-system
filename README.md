# Music-Recommendation-system

!pip install spotipy

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up your Spotify API credentials
client_id = '14355fba52df430cab7d441c6df8396b'
client_secret = '9e8758a735634f099bb325f1f94d3927'

# Set up the Spotipy client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define a search query to find tracks
query = ['Emiway Bantai']


# Set the tracks 
max_tracks = 50

# Search for tracks
results = sp.search(q=query, type='track', limit=max_tracks)

# Extract the track IDs from the search results
track_ids = [track['id'] for track in results['tracks']['items']]

# Print the track IDs
print(track_ids)


def collect_track_data(track_id):
    track_data = sp.track(track_id)
    artists = ', '.join([artist['name'] for artist in track_data['artists']])
    album_name = track_data['album']['name']
    track_name = track_data['name']
    popularity = track_data['popularity']
    duration_ms = track_data['duration_ms']
    audio_features = sp.audio_features(track_id)[0]
    danceability = audio_features['danceability']
    energy = audio_features['energy']
    key = audio_features['key']
    loudness = audio_features['loudness']
    mode = audio_features['mode']
    speechiness = audio_features['speechiness']
    acousticness = audio_features['acousticness']
    instrumentalness = audio_features['instrumentalness']
    liveness = audio_features['liveness']
    valence = audio_features['valence']
    tempo = audio_features['tempo']
    time_signature = audio_features['time_signature']

    return [artists, album_name, track_name, popularity, duration_ms, danceability,energy  ,key, loudness, mode,
            speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature]
            
# Initialize the collected data list
collected_data = []

# Collect data for each track ID
for track_id in track_ids:
    collected_data.append(collect_track_data(track_id))
    
for data in collected_data:
    print(data)
    
column_names = ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'danceability',
           'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
           'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
           
 for row in data:
    if len(row) != len(column_names):
        continue  # Skip the row if its length doesn't match column names
        
    # Process the row
    print("Processing row:", row)
    
import pandas as pd
data = pd.DataFrame(row, columns=column)

data.to_csv('spotify_rap_data.csv')

#Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


# Load the dataset into a pandas DataFrame
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\spotify\spotify_rap_data.csv")

# Trimming dataset
data = data.iloc[:,1:]

data.head()
data.info()

# checking for missing value
data.isna().sum()

# checking for duplicate value
data.duplicated().sum()

# drop duplicate value
data.drop_duplicates()

# Feature Correlation
correlation_matrix = data.corr()

# visualisation of corealtion
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scale Normalization
numeric_columns = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                   'time_signature']

scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
# K-means clustering
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data[numeric_columns])
    TWSS.append(kmeans.inertia_)
    
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(data[numeric_columns])

# Find Similarity
similarity_matrix = cosine_similarity(data[numeric_columns])

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[numeric_columns])
data['PC1'] = principal_components[:, 0]
data['PC2'] = principal_components[:, 1]

# Visualization of clustering and PCA results
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=data['cluster'])
plt.show()

# Recommendation Function

def get_recommendations(artist_name):
    # Filter data for the given artist
    artist_data = data[data['artists'].str.contains(artist_name, case=False)]

    if artist_data.empty:
        return "No recommendations available for the given artist."

    # Find the mean popularity for each album
    album_popularity = artist_data.groupby('album_name')['popularity'].mean()

    # Find the most popular albums
    top_albums = album_popularity.nlargest(5).index.tolist()

    # Find the most similar albums
    artist_index = artist_data.index.tolist()[0]
    similarity_scores = similarity_matrix[artist_index]
    similar_albums_indices = np.argsort(similarity_scores)[::-1][1:6]
    similar_albums = data.loc[similar_albums_indices, 'album_name'].tolist()

    return top_albums ,similar_albums
    
# Recommendation
artist_name = input('\nEnter artist name:  ')
top_albums , similar_albums = get_recommendations(artist_name)

print("\nTOP 5 ALBUMS: \n")
for album in top_albums:
    print(album)
    
print("\nSIMILAR ALBUMS: \n")
for album in similar_albums:
    print(album)
