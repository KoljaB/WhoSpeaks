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

- **fetch_youtube_mp3.py**: Extracts and converts YouTube audio, like podcasts, to MP3 for voice analysis.
- **split_dataset.py**: This tool divides your input audio into distinct sentences.
- **convert_wav.py**: Converts the sentence-based MP3 files into WAV format.
- **speaker_diarize.py**: Heart of WhoSpeaks. Categorizes sentences into speaker groups and selects training sentences based on the unique algorithm described above.

> **Note**: *current implementation is for two speakers*

I initially developed this as a personal project, but was astounded by its effectiveness. In my first tests it outperformed existing solutions like pyannote audio in both reliability and speed while being the more lightweight approach. For me it could be a significant step up in voice diarization capabilities, that's why I've decided to release this rather raw, yet powerful code for others to experiment with.

# WhoSpeaks

*A Toolkit for Enhanced Voice Training Datasets*

WhoSpeaks is an innovative project designed to surpass existing speaker diarization tools in reliability, speed, and efficiency. Born from a need for more effective speaker diarization, WhoSpeaks offers a unique approach to voice analysis and sentence categorization. 

## Core Concept

1. **Voice Characteristic Extraction**: Unique voice characteristics are extracted from each sentence in an audio file, creating distinct audio embeddings.
2. **Sentence Similarity Comparison**: Cosine similarity is utilized to compare these embeddings, identifying similarities between sentences.
3. **Grouping and Averaging**: Similar sounding sentences are grouped, and anomalies are averaged out, minimizing errors.
4. **Identification of Distinct Groups**: Analysis of these groups helps isolate unique speaker characteristics, allowing precise speaker matching.

## Feature Modules

- **fetch_youtube_mp3.py**: Downloads YouTube audio, such as podcasts, and converts it to MP3 for voice analysis.
- **split_dataset.py**: Splits input audio into distinct sentences.
- **convert_wav.py**: Converts MP3 files into WAV format.
- **speaker_diarize.py**: The core of WhoSpeaks, categorizing sentences into speaker groups based on the unique algorithm.
- **pyannote_diarize.py**: Use for comparison against pyannote audio, a current state of the art speaker diarization model

*Note: The current implementation is optimized for two speakers.*

## Performance and Testing

To demonstrate WhoSpeaks' capabilities, we conducted a test using a challenging audio sample: the 4:38 Coin Toss scene from "No Country for Old Men". In this scene, the two male speakers have very similar voice profiles, presenting a difficult scenario for diarization libraries.

### Process:

1. **Download**: Using `fetch_youtube_mp3.py`, we downloaded the MP3 from the scene's YouTube video.
2. **Diarization Comparison**: We first ran the scene through `pyannote_diarize.py` (from pyannote audio) and set the speaker parameters to 2.
   - Pyannote's output was inaccurate, assigning most sentences to one speaker incorrectly.
3. **WhoSpeaks Analysis**: 
   - **Sentence Splitting**: We used `split_dataset.py` with `tiny.en` for efficiency, though `large-v2` offers higher accuracy.
   - **Conversion**: The MP3 segments were converted to WAV format using `convert_wav.py`.
   - **Diarization**: We then ran `speaker_diarize.py` and visually inspected the dendrogram file to confirm the presence of two speakers.

### Results:

- WhoSpeaks' algorithm assigned 53 sentences correctly to Javier Bardem's voice with only 2 minor errors.
- Of the 33 sentences assigned to the other actor, only one was incorrect.
- The overall error rate was approximately 3.5%, demonstrating a precision of about 95% in correctly assigning sentences.

The effectiveness of WhoSpeaks in this test, particularly against pyannote audio, showcases its potential in handling complex diarization scenarios with high accuracy and efficiency. This project, initially a personal endeavor, has evolved into a powerful tool for voice diarization, offering a lightweight yet sophisticated alternative to existing solutions. We encourage others to experiment with this approach and contribute to its development.