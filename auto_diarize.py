print("Auto Speaker Diarization with Hierarchical Clustering")

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config import load_config
import matplotlib.pyplot as plt
import numpy as np
import librosa
import shutil
import torch
import os

# Setup
input_directory = 'output_sentences_wav'
output_directory = 'output_speakers'
max_sentences = 1000
minimum_duration = 0.5
two_speaker_threshold = 19
silhouette_diff_threshold = 0.01  # Adjust as needed for your data
data = []

print("Loading TTS model")
device = torch.device("cuda")
local_models_path = os.environ.get("COQUI_MODEL_PATH")
if local_models_path is None:
    local_models_path = "models"
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


def get_speaker_embedding(audio_path):
    _, speaker_embedding = tts.get_conditioning_latents(audio_path=audio_path, gpt_cond_len=30, max_ref_length=60)
    return speaker_embedding


# Create 1D embeddings from sentences
count = 0
embeddings = []
for filename in os.listdir(input_directory):
    if filename.endswith(".wav") and count < max_sentences:
        y, sr = librosa.load(os.path.join(input_directory, filename))
        if librosa.get_duration(y=y, sr=sr) >= minimum_duration:
            full_path = os.path.join(input_directory, filename)
            speaker_embedding = get_speaker_embedding(full_path)
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

# Safety check using KMeans for initial speaker detection
def determine_optimal_cluster_count(embeddings_scaled):
    num_embeddings = len(embeddings_scaled)
    if num_embeddings <= 1:
        # Only one embedding, so only one speaker
        return 1
    else:
        # Determine single or multiple speakers
        # K-means Clustering with k=2
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
        distances = kmeans.transform(embeddings_scaled)
        avg_distance = np.mean(np.min(distances, axis=1))
        distance_threshold = two_speaker_threshold  # Threshold to decide if we have one or multiple speakers

        if avg_distance < distance_threshold:
            print(f"Single Speaker: low embedding distance: {avg_distance} < {distance_threshold}.")
            return 1
        else:
            # Hierarchical Clustering for multiple speakers
            max_clusters = min(10, num_embeddings)
            range_clusters = range(2, max_clusters + 1)
            silhouette_scores = []

            for n_clusters in range_clusters:
                hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                cluster_labels = hc.fit_predict(embeddings_scaled)

                unique_labels = set(cluster_labels)
                if 1 < len(unique_labels) < len(embeddings_scaled):
                    silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    print(f"Inappropriate number of clusters: {len(unique_labels)}.")
                    silhouette_scores.append(-1)

            # Find the optimal number of clusters based on silhouette scores
            optimal_cluster_count = 2
            for i in range(1, len(silhouette_scores)):
                # Ensure a significant increase in the silhouette score to add a new cluster
                if silhouette_scores[i] - silhouette_scores[i - 1] > silhouette_diff_threshold:
                    optimal_cluster_count = range_clusters[i]
                # else:
                #     print(f"Silhouette score difference too low: {silhouette_scores[i] - silhouette_scores[i - 1]}.")

            # optimal_cluster_count = range_clusters[silhouette_scores.index(max(silhouette_scores))]

            return optimal_cluster_count

# Determine the optimal number of clusters
optimal_cluster_count = determine_optimal_cluster_count(embeddings_scaled)

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')

# Save or show dendrogram
dendrogram_file = 'dendrogram.png'
plt.savefig(dendrogram_file)
print()
print(f"The dendrogram image showing the detected speaker clusters was saved as {dendrogram_file}.")
print()

# Explanation for the user
print(f"The automatical speaker detection suggested there were {optimal_cluster_count} speakers.")
print(f"Please verify this by inspecting the dendrogram plot that has been saved as {dendrogram_file}.")
print("You should look for the longest vertical lines that are not crossed by any horizontal lines.")
print("These lines suggest a natural separation between different clusters.")
print("A horizontal 'cut' through these long lines will determine the number of clusters.")
print("Count the number of vertical lines intersected by an imaginary horizontal line to decide the cluster count.")
print("This number will be the number of speakers you should input.")
print()
print(f"Automatical speaker count suggestion: {optimal_cluster_count} speakers.")
print()
print("If you have identified a different number of speakers from the dendragram file, please enter the number.")
print(f"If you are satisfied with the automatic suggestion of {optimal_cluster_count} speakers, you can press Enter to proceed.")
print()

# Ask the user for the number of clusters with a retry mechanism if the input fails
while True:
    try:
        input_user = input("Please enter the number of speakers (clusters) you have identified: ")
        if input_user == "":
            cluster_count = optimal_cluster_count
            break
        cluster_count = int(input_user)
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
