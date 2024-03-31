# WhoSpeaks

*A Toolkit for Enhanced Voice Training Datasets*

I created WhoSpeaks out of frustration with the current speaker diarization libraries. My experience found them to be generally unreliable and too heavyweight. WhoSpeaks aims to offer a more precise and user-friendly alternative for voice training datasets.

I initially crafted this as a personal project, but was astounded by its effectiveness. For me it's a significant step up from other tools like pyannote audio. That's why I've decided to release this raw, yet powerful code for others to experiment with and enhance their diarization processes.

Here's the core concept:
- **Voice Characteristic Extraction**: For each sentence in your audio, unique voice characteristics are extracted, creating audio embeddings.
- **Sentence Similarity Comparison**: Then cosine similarity is used to compare these embeddings against every other sentence, identifying similarities.
- **Grouping and Averaging**: Similar sounding sentences are grouped together. This approach averages out anomalies and minimizes errors from individual data points.
- **Identification of Distinct Groups**: By analyzing these groups, we can isolate the most distinct ones, which represent unique speaker characteristics.

These steps allow us to match any sentence against the established speaker profiles with remarkable precision.

### Feature Modules

- **split_dataset.py**: This tool divides your input audio into distinct sentences.
- **convert_wav.py**: Converts the sentence-based MP3 files into WAV format.
- **speaker_diarize.py**: Heart of WhoSpeaks. Categorizes sentences into speaker groups and selects training sentences based on the unique algorithm described above.