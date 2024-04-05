import os
import numpy as np
import shutil
import librosa
import torch
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config import load_config
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Setup
input_directory = 'output_sentences_wav'
output_directory = 'output_speakers'
max_sentences = 1000
minimum_duration = 0.5

data = []
print("Loading TTS model")
device = torch.device("cuda")
local_models_path = os.environ.get("COQUI_MODEL_PATH")
checkpoint = os.path.join(local_models_path, "v2.0.2")
config = load_config((os.path.join(checkpoint, "config.json")))
tts = setup_tts_model(config)
tts.load_checkpoint(
    config,
    checkpoint_dir=checkpoint,
    checkpoint_path=None,
    vocab_path=None,
    eval=True,
    use_deepspeed=False,
)
tts.to(device)
print("TTS model loaded")

# Create 1D embeddings from sentences
count = 0
embeddings = []
for filename in os.listdir(input_directory):
    if filename.endswith(".wav") and count < max_sentences:
        y, sr = librosa.load(os.path.join(input_directory, filename))
        if librosa.get_duration(y=y, sr=sr) >= minimum_duration:
            full_path = os.path.join(input_directory, filename)
            _, speaker_embedding = tts.get_conditioning_latents(audio_path=full_path, gpt_cond_len=30, max_ref_length=60)
            speaker_embedding_1D = speaker_embedding.view(-1).cpu().detach().numpy()
            embeddings.append(speaker_embedding_1D)
            data.append({'filename': filename, 'speaker_embeds_1D': speaker_embedding_1D})
            count += 1


# Standard scaling
embeddings_array = np.array(embeddings)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_array)

# Hierarchical Clustering
linked = sch.linkage(embeddings_scaled, method='ward')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')

# Save or show dendrogram
dendrogram_file = 'dendrogram.png'
plt.savefig(dendrogram_file)
print(f"Dendrogram saved as {dendrogram_file}")

# Explanation for the user
print(f"Please inspect the dendrogram plot that has been saved as {dendrogram_file}.")
print("You should look for the longest vertical lines that are not crossed by any horizontal lines.")
print("These lines suggest a natural separation between different clusters.")
print("A horizontal 'cut' through these long lines will determine the number of clusters.")
print("Count the number of vertical lines intersected by an imaginary horizontal line to decide the cluster count.")
print("This number will be the number of speakers you should input.")

# Ask the user for the number of clusters with a retry mechanism if the input fails
while True:
    try:
        cluster_count = int(input("Enter the number of speakers (clusters) you have identified: "))
        if cluster_count > 0:
            break
        else:
            print("The number of clusters must be a positive integer. Please try again.")
    except ValueError:
        print("Invalid input; please enter an integer value. Try again.")


# Determine clusters from dendrogram
hc = AgglomerativeClustering(n_clusters=cluster_count, linkage='ward')
clusters = hc.fit_predict(embeddings_scaled)

# Assign sentences to clusters
for i, entry in enumerate(data):
    entry['assigned_cluster'] = clusters[i]

# Copy files to corresponding directories
for cluster_id in range(cluster_count):
    speaker_dir = os.path.join(output_directory, f"speaker_{cluster_id}")
    os.makedirs(speaker_dir, exist_ok=True)

    for entry in data:
        if entry['assigned_cluster'] == cluster_id:
            source_path = os.path.join(input_directory, entry['filename'])
            destination_path = os.path.join(speaker_dir, entry['filename'])
            shutil.copy(source_path, destination_path)

print("Speaker diarization completed with hierarchical clustering.")




# import os
# import numpy as np
# import shutil
# import librosa
# import torch
# from TTS.tts.models import setup_model as setup_tts_model
# from TTS.config import load_config
# from scipy.spatial.distance import cosine
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # Setup
# input_directory = 'output_sentences_wav'
# output_directory = 'output_speakers'
# max_sentences = 1000
# minimum_duration = 4

# data = []
# print("Loading TTS model")
# device = torch.device("cuda")
# local_models_path = os.environ.get("COQUI_MODEL_PATH")
# checkpoint = os.path.join(local_models_path, "v2.0.2")
# config = load_config((os.path.join(checkpoint, "config.json")))
# tts = setup_tts_model(config)
# tts.load_checkpoint(
#     config,
#     checkpoint_dir=checkpoint,
#     checkpoint_path=None,
#     vocab_path=None,
#     eval=True,
#     use_deepspeed=False,
# )
# tts.to(device)
# print("TTS model loaded")

# # Create 1D embeddings from sentences
# count = 0
# embeddings = []
# for filename in os.listdir(input_directory):
#     if filename.endswith(".wav") and count < max_sentences:
#         y, sr = librosa.load(os.path.join(input_directory, filename))
#         if librosa.get_duration(y=y, sr=sr) >= minimum_duration:
#             full_path = os.path.join(input_directory, filename)
#             _, speaker_embedding = tts.get_conditioning_latents(audio_path=full_path, gpt_cond_len=30, max_ref_length=60)
#             speaker_embedding_1D = speaker_embedding.view(-1).cpu().detach().numpy()
#             embeddings.append(speaker_embedding_1D)
#             data.append({'filename': filename, 'speaker_embeds_1D': speaker_embedding_1D})
#             count += 1

# # Clustering using DBSCAN
# embeddings_array = np.array(embeddings)
# scaler = StandardScaler()
# embeddings_scaled = scaler.fit_transform(embeddings_array)

# # DBSCAN parameters to try
# eps_values = np.arange(0.1, 1.1, 0.1)  # From 0.1 to 1.0 in steps of 0.1
# min_samples_values = range(2, 11)  # From 2 to 10 in steps of 1

# found_clusters = False
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         clusters = dbscan.fit_predict(embeddings_scaled)

#         if len(set(clusters)) > 1 and -1 not in set(clusters):
#             found_clusters = True
#             print(f"Clusters found with eps={eps} and min_samples={min_samples}")
#             break
#     if found_clusters:
#         break

# if not found_clusters:
#     print("No distinct clusters found with tried parameters. Consider alternative clustering methods.")
#     exit()


# # Calculate cluster centroids
# centroids = []
# for cluster_id in set(clusters):
#     if cluster_id != -1:  # Exclude outliers
#         cluster_embeddings = embeddings_array[clusters == cluster_id]
#         centroid = np.mean(cluster_embeddings, axis=0)
#         centroids.append(centroid)

# # Check if centroids list is empty
# if not centroids:
#     print("No centroids found. All data might be classified as outliers.")
#     exit()

# # Assign sentences to clusters
# for entry in data:
#     embedding = entry['speaker_embeds_1D']
#     distances = [cosine(embedding, centroid) for centroid in centroids]
#     assigned_cluster = np.argmin(distances)
#     entry['assigned_cluster'] = assigned_cluster

# # Create output directories and copy files
# for cluster_id in range(len(centroids)):
#     speaker_dir = os.path.join(output_directory, f"speaker_{cluster_id}")
#     os.makedirs(speaker_dir, exist_ok=True)

#     for entry in data:
#         if entry['assigned_cluster'] == cluster_id:
#             source_path = os.path.join(input_directory, entry['filename'])
#             destination_path = os.path.join(speaker_dir, entry['filename'])
#             shutil.copy(source_path, destination_path)

# print("Speaker diarization completed.")
