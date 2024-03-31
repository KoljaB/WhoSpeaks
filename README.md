# WhoSpeaks

*A Toolkit for Enhanced Voice Training Datasets*

WhoSpeaks emerged from the need for better speaker diarization tools. Existing libraries are heavyweight and often fall short in reliability, speed and efficiency. So this project offers a more refined alternative.

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

> **Note**: *current implementation is for two speakers*

I initially developed this as a personal project, but was astounded by its effectiveness. In my first tests it outperformed existing solutions like pyannote audio in both reliability and speed while being the more lightweight approach. For me it could be a significant step up in voice diarization capabilities, that's why I've decided to release this rather raw, yet powerful code for others to experiment with.
