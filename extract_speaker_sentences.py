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
speaker_reference_file = '0035093-0035300_25_so_it_s_easier_.wav' # no path
max_sentences = 1000000
group_percentage = 0.1
minimum_duration = 1
only_keep_most_confident_percentage = 0.8

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
speaker_embeddings = None
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

        if speaker_reference_file == filename:
            print(f"Speaker reference found: {filename}")
            speaker_embeddings = speaker_embedding_1D

        entry = {
            'filename': filename,
            'speaker_embeds_1D': speaker_embedding_1D
        }
        data.append(entry)
    else:
        continue

if speaker_embeddings is None:
    raise Exception("Speaker reference not found")

# Check similarity of each sentence to the speaker reference
for index, entry in enumerate(data):
    embedding = entry['speaker_embeds_1D']
    similarity = 1 - cosine(embedding, speaker_embeddings)
    entry['confidence'] = similarity

# Sort the data by confidence
data.sort(key=lambda x: x['confidence'], reverse=True)

# Create subdirectories for each percentile
percentile_directories = []
for i in range(10):
    dir_name = os.path.join(output_directory, f'percentile_{i * 10}-{(i + 1) * 10}')
    os.makedirs(dir_name, exist_ok=True)
    percentile_directories.append(dir_name)


# Assign each file to its percentile directory
total_files = len(data)
for index, entry in enumerate(data):
    percentile_index = (index * 10) // total_files  # Find the correct percentile
    destination_dir = percentile_directories[percentile_index]
    base_name, extension = os.path.splitext(entry['filename'])
    new_filename = f"{base_name}_conf_{entry['confidence']:.2f}{extension}"  # Append confidence to filename
    source_path = os.path.join(input_directory, entry['filename'])
    destination_path = os.path.join(destination_dir, new_filename)

    # Copy the file to the percentile directory with new filename
    shutil.copy(source_path, destination_path)
    print(f"Copied {entry['filename']} to {destination_path}")
