"""
Speaker Diarization

Idea:
 - create 1D embeddings specs from sentences
 - for every sentence
    - find most similar 10% other sentences
    - average out the 1Ds and make a "speech group" embedding from that
 - for every sentence
    - compare speech group embedding with all other sentence speech group embeddings
    - find the two speech groups with least similar embeddings
    - the 2 "speech group" embedding from that will be our "speaker" characteristics 1D embeddings
 - for every sentence
    - find cosine similarity between the sentence and the two "speaker" characteristics 1D embeddings
    - assign to the speaker with higher similarity

=> every sentence assigned to one to two speakers

notes:
- cut out every < 3s file before processing

"""

from TTS.tts.models import setup_model as setup_tts_model
from scipy.spatial.distance import cosine
from TTS.config import load_config
import librosa.display
import librosa
import numpy as np
import shutil
import torch
import os

input_directory = 'output_sentences_wav'
output_directory = 'output_speakers'
max_sentences = 1000000
group_percentage = 0.1
minimum_duration = 4
only_keep_most_confident_percentage = 0.5

data = []

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

# create 1D embeddings from sentences
count = 0
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        count += 1
        if count > max_sentences:
            break

        # skip if file is too short
        y, sr = librosa.load(os.path.join(input_directory, filename))
        if librosa.get_duration(y=y, sr=sr) < minimum_duration:
            continue

        full_path = os.path.join(input_directory, filename)
        print(full_path)

        gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(audio_path=full_path, gpt_cond_len=30, max_ref_length=60)
        spealer_embedding = speaker_embedding.cpu().squeeze().half().tolist()
        speaker_embedding_1D = speaker_embedding.view(-1).cpu().detach().numpy()  # Reshape to 1D then convert to NumPy

        entry = {
            'filename': filename,
            'speaker_embeds_1D': speaker_embedding_1D
        }
        data.append(entry)
    else:
        continue

# Find most similar 10% other sentences
# Calculate 10% of the number of sentences, at least 1    
num_top_sentences = max(1, int(group_percentage * len(data)))
print(f"Sentences per group: {num_top_sentences}")

# Find speech group embedding of sentence
for index, entry in enumerate(data):
    similarities = []
    embedding = entry['speaker_embeds_1D']

    # Compute similarities with other sentences
    for index_compare, compare_entry in enumerate(data):
        if index_compare != index:
            embedding_compare = compare_entry['speaker_embeds_1D']
            similarity = 1 - cosine(embedding, embedding_compare)
            similarities.append((similarity, embedding_compare))

    # Sort by similarity and pick top 10%
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_similar_embeddings = [x[1] for x in similarities[:num_top_sentences]]

    # Step 2: Average out the 1Ds and make a "speech group" embeddinngs
    speech_group_embedding = np.mean(np.array(top_similar_embeddings), axis=0)

    # Step 3: Store the speech group embedding in data
    entry['speech_group_embed'] = speech_group_embedding

# Find speakers by comparing speech group embeddings
for index, entry in enumerate(data):
    similarities = []
    embedding = entry['speech_group_embed']
    for index_compare, compare_entry in enumerate(data):
        if index_compare != index:
            embedding_compare = compare_entry['speech_group_embed']
            similarity = 1 - cosine(embedding, embedding_compare)
            similarities.append((similarity, embedding_compare))

    # Sort by similarity and pick least similar
    similarities.sort(reverse=False, key=lambda x: x[0])
    least_similar_embed = similarities[0][1]
    entry['least_similarity'] = similarities[0][0]
    entry['least_similar_embed'] = least_similar_embed

# Find entry with least similarity
data.sort(reverse=False, key=lambda x: x['least_similarity'])
least_similar_entry = data[0]

embed_speaker_1 = least_similar_entry['speech_group_embed']
embed_speaker_2 = least_similar_entry['least_similar_embed']

for entry in data:
    similarity_1 = 1 - cosine(entry['speaker_embeds_1D'], embed_speaker_1)
    similarity_2 = 1 - cosine(entry['speaker_embeds_1D'], embed_speaker_2)

    if similarity_1 > similarity_2:
        entry['speaker'] = 1
        entry['confidence'] = similarity_1 - similarity_2
    else:
        entry['speaker'] = 2
        entry['confidence'] = similarity_2 - similarity_1

    print(f"Speaker {entry['speaker']} assigned to {entry['filename']} with confidence {entry['confidence']}")

# Remove the least confident
data.sort(reverse=True, key=lambda x: x['confidence'])
data = data[:int(len(data) * only_keep_most_confident_percentage)]

# Ensure output directories exist
for speaker_id in [1, 2]:
    speaker_dir = os.path.join(output_directory, f"speaker_{speaker_id}")
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)

# Copy files to the corresponding speaker directory
for entry in data:
    speaker = entry['speaker']
    filename = entry['filename']
    source_path = os.path.join(input_directory, filename)
    destination_path = os.path.join(output_directory, f"speaker_{speaker}", filename)

    # Copy the file
    shutil.copy(source_path, destination_path)
    print(f"Copied {filename} to {destination_path}")
