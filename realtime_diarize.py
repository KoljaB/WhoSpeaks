from PyQt6.QtWidgets import QApplication, QTextEdit, QMainWindow
from PyQt6.QtCore import pyqtSignal, QThread, QEvent, QTimer
from sklearn.cluster import AgglomerativeClustering, KMeans
from TTS.tts.models import setup_model as setup_tts_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from RealtimeSTT import AudioToTextRecorder
from scipy.spatial.distance import cosine
import scipy.cluster.hierarchy as sch
from TTS.config import load_config
import numpy as np
import pyaudio
import queue
import torch
import wave
import sys
import os

SILENCE_THRESHS = [0, 0.4]
FAST_SENTENCE_END = True
LOOPBACK_DEVICE_NAME = "stereomix"
LOOPBACK_DEVICE_HOST_API = 0

# TWO_SPEAKER_THRESHOLD = 0.8
TWO_SPEAKER_THRESHOLD = 19
SILHOUETTE_DIFF_THRESHOLD = 0.01  # Adjust as needed for your data

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
BUFFER_SIZE = 512

COLOR_TABLE_HEX = [
    "#FFFF00",  # yellow
    "#FF0000",  # red
    "#00FFFF",  # cyan
    "#FF00FF",  # magenta
    "#FFA500",  # orange
    "#00FF00",  # lime
    "#800080",  # purple
    "#FFC0CB",  # pink
    "#008080",  # teal
    "#FF7F50",  # coral
    "#00FFFF",  # aqua
    "#8A2BE2",  # violet
    "#FFD700",  # gold
    "#7FFF00",  # chartreuse
    "#FF00FF",  # fuchsia
    "#A0522D",  # sienna
    "#40E0D0",  # turquoise
    "#D2691E",  # chocolate
    "#DC143C",  # crimson
    "#FA8072",  # salmon
    "#DA70D6",  # orchid
    "#DDA0DD",  # plum
    "#FFBF00",  # amber
    "#007FFF",  # azure
    "#F5F5DC",  # beige
    "#E6E6FA",  # lavender
    "#CC7722",  # ochre
    "#FFDAB9",  # peach
    "#9B111E",  # ruby
    "#C0C0C0",  # silver
    "#D2B48C",  # tan
    "#F5DEB3",  # wheat
    "#CD7F32",  # bronze
    "#3EB489",  # mint
    "#EAE0C8",  # pearl
    "#0F52BA",  # sapphire
    "#F28500",  # tangerine
    "#50C878",  # emerald
    "#FF007F",  # rose
    "#9966CC",  # amethyst
    "#2A52BE",  # cerulean
    "#B87333",  # copper
    "#FFFFF0",  # ivory
    "#C3B091",  # khaki
    "#E30B5D",  # raspberry
    "#D9381E",  # vermilion
    "#36454F",  # charcoal
    "#FC8EAC",  # flamingo
    "#00A36C",  # jade
    "#FFF44F",  # lemon
    "#D9E650",  # quartz
    "#FF6347",  # tomato
    "#0047AB",  # cobalt
    "#F4C430",  # saffron
    "#F9543B",  # zinnia
    "#808000",  # olive
    "#800000",  # maroon
    "#000080",  # navy
    "#008000",  # green
    "#0000FF",  # blue
    "#800000",  # merlot
    "#4B0082",  # indigo
]


class TextRetrievalThread(QThread):
    textRetrievedFinal = pyqtSignal(str, np.ndarray)
    textRetrievedLive = pyqtSignal(str)
    recorderStarted = pyqtSignal()

    def __init__(self):
        super().__init__()

    def live_text_detected(self, text):
        self.textRetrievedLive.emit(text)

    def run(self):
        print("Emitted Starting recorder")
        recorder_config = {
            'spinner': False,
            'use_microphone': False,
            'model': 'large-v3',
            'language': 'en',
            'silero_sensitivity': 0.4,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': SILENCE_THRESHS[1],
            'min_length_of_recording': 0.7,
            'pre_recording_buffer_duration': 0.35,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': 'distil-small.en',
            'on_realtime_transcription_update': self.live_text_detected,
            'beam_size': 5,
            'beam_size_realtime': 5,
            'buffer_size': BUFFER_SIZE,
            'sample_rate': SAMPLE_RATE,
        }

        self.recorder = AudioToTextRecorder(**recorder_config)
        self.recorderStarted.emit()

        def process_text(text, bytes):
            self.textRetrievedFinal.emit(text, bytes)

        while True:
            self.recorder.text(process_text)


class TextUpdateThread(QThread):
    text_update_signal = pyqtSignal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        self.text_update_signal.emit(self.text)


class RecordingThread(QThread):
    def __init__(self, recorder):
        super().__init__()

        self.audio = pyaudio.PyAudio()

        def find_stereo_mix_index():
            devices = ""
            for i in range(self.audio.get_device_count()):
                dev = self.audio.get_device_info_by_index(i)
                devices += f"{dev['index']}: {dev['name']} "
                f"(hostApi: {dev['hostApi']})\n"

                if (LOOPBACK_DEVICE_NAME in dev['name'].lower()
                        and dev['hostApi'] == LOOPBACK_DEVICE_HOST_API):
                    return dev['index'], devices

            return None, devices

        stereo_mix_index, devices = find_stereo_mix_index()

        if stereo_mix_index is None:
            print("Stereo Mix device not found")
            print("Available devices:\n", devices)
            self.audio.terminate()
            exit()
        else:
            print(f"Stereo Mix device found at index: {stereo_mix_index}")

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE,
            input_device_index=stereo_mix_index)
        self.recorder = recorder
        self._is_running = True

    def run(self):
        while self._is_running:
            data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
            self.recorder.feed_audio(data)

    def stop(self):
        self._is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


class SentenceWorker(QThread):
    sentence_update_signal = pyqtSignal(list, list)

    def __init__(self, queue, tts_model):
        super().__init__()
        self.queue = queue
        self.tts = tts_model
        self._is_running = True
        self.full_sentences = []
        self.sentence_speakers = []
        self.speaker_index = 0
        self.speakers = []

    def run(self):
        while self._is_running:
            try:
                text, bytes = self.queue.get(timeout=1)
                self.process_item(text, bytes)
            except queue.Empty:
                continue

    def process_speakers(self):
        embeddings = [speaker_embedding for _, speaker_embedding in self.full_sentences]

        # Standard scaling
        embeddings_array = np.array(embeddings)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)

        num_embeddings = len(embeddings_scaled)
        if num_embeddings <= 1:
            # Only one embedding, so only one speaker
            optimal_cluster_count = 1
        else:
            # Determine single or multiple speakers
            # K-means Clustering with k=2
            kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
            distances = kmeans.transform(embeddings_scaled)
            avg_distance = np.mean(np.min(distances, axis=1))
            distance_threshold = TWO_SPEAKER_THRESHOLD

            # Determine Single or Multiple Speakers
            if avg_distance < distance_threshold:
                optimal_cluster_count = 1  # Only one speaker
                print(f"1 Speaker: low embedding distance: {avg_distance} < {distance_threshold}.")
            else:
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
                    if silhouette_scores[i] - silhouette_scores[i - 1] > SILHOUETTE_DIFF_THRESHOLD:
                        optimal_cluster_count = range_clusters[i]
                    else:
                        print(f"Silhouette score difference too low: {silhouette_scores[i] - silhouette_scores[i - 1]}.")

                print(f"{optimal_cluster_count} Speakers: high embedding distance: {avg_distance} > {distance_threshold}.")

        if optimal_cluster_count == 1:
            self.sentence_speakers = [0] * len(self.full_sentences)
        else:
            self.sentence_speakers = []

            # Determine clusters
            hc = AgglomerativeClustering(n_clusters=optimal_cluster_count, linkage='ward')
            clusters = hc.fit_predict(embeddings_scaled)

            # Assign sentences to clusters
            # Create a mapping from old to new cluster indices
            cluster_mapping = {}
            new_index = 0
            for cluster in clusters:
                if cluster not in cluster_mapping:
                    cluster_mapping[cluster] = new_index
                    new_index += 1

            # Assign sentences to clusters with new indices
            for cluster in clusters:
                self.sentence_speakers.append(cluster_mapping[cluster])

    def process_item(self, text, bytes):
        audio_int16 = np.int16(bytes * 32767)

        tempfile = "output.wav"
        with wave.open(tempfile, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())

        for tries in range(3):
            try:
                _, speaker_embedding = self.tts.get_conditioning_latents(
                    audio_path=tempfile,
                    gpt_cond_len=30,
                    max_ref_length=60)
                speaker_embedding = \
                    speaker_embedding.view(-1).cpu().detach().numpy()
                break
            except Exception as e:
                print(f"Error in try {tries}: {e}")
                speaker_embedding = np.zeros(512)

        self.full_sentences.append((text, speaker_embedding))
        self.process_speakers()
        self.sentence_update_signal.emit(self.full_sentences, self.sentence_speakers)

    def stop(self):
        self._is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Realtime Speaker Diarization")

        self.tts = None
        self.initialized = False
        self.displayed_text = ""
        self.last_realtime_text = ""
        self.full_sentences = []
        self.sentence_speakers = []
        self.speaker_index = 0
        self.pending_sentences = []
        # self.speakers = []
        self.queue = queue.Queue()

        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet("""
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Arial';
            font-size: 16pt;
        """)
        self.setCentralWidget(self.text_edit)

    def showEvent(self, event):
        super().showEvent(event)
        if event.type() == QEvent.Type.Show:
            if not self.initialized:
                self.initialized = True
                self.resize(1200, 800)
                self.update_text("Please wait until app is loaded")

                QTimer.singleShot(500, self.init)

    def process_live_text(self, text):
        text = text.strip()

        if text:
            sentence_delimiters = '.?!。'
            prob_sentence_end = (
                len(self.last_realtime_text) > 0
                and text[-1] in sentence_delimiters
                and self.last_realtime_text[-1] in sentence_delimiters
            )

            self.last_realtime_text = text

            if prob_sentence_end:
                if FAST_SENTENCE_END:
                    self.text_retrieval_thread.recorder.stop()
                else:
                    self.text_retrieval_thread.recorder.post_speech_silence_duration = SILENCE_THRESHS[0]
            else:
                self.text_retrieval_thread.recorder.post_speech_silence_duration = SILENCE_THRESHS[1]

            self.last_realtime_text = text

        self.text_detected(text)

    def text_detected(self, text):

        try:
            sentences_with_style = []
            for i, sentence in enumerate(self.full_sentences):
                sentence_text, speaker_embedding = sentence
                sentence_tshort = sentence_text[:40]
                if i >= len(self.sentence_speakers):
                    print(f"Index {i} out of range")
                    color = "#FFFFFF"
                else:
                    speaker_index = self.sentence_speakers[i]
                    color = COLOR_TABLE_HEX[speaker_index % len(COLOR_TABLE_HEX)]

                sentences_with_style.append(
                    f'<span style="color:{color};">{sentence_text}</span>')

            for pending_sentence in self.pending_sentences:
                sentences_with_style.append(
                    f'<span style="color:#60FFFF;">{pending_sentence}</span>')

            new_text = " ".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

            if new_text != self.displayed_text:
                self.displayed_text = new_text
                self.update_text(new_text)
        except Exception as e:
            print(f"Error: {e}")


    def process_final(self, text, bytes):
        text = text.strip()
        if text:
            try:
                self.pending_sentences.append(text)
                self.queue.put((text, bytes))
            except Exception as e:
                print(f"Error: {e}")

    def recording_thread(self, stream):
        while True:
            data = stream.read(BUFFER_SIZE)
            self.text_retrieval_thread.recorder.feed(data)

    def capture_output_and_feed_to_recorder(self):
        self.recording_thread = RecordingThread(
            self.text_retrieval_thread.recorder)
        self.recording_thread.start()

    def recorder_ready(self):
        print("Recorder ready")
        self.update_text("Ready to record")

        self.capture_output_and_feed_to_recorder()

    def init(self):
        self.start_tts()

        print("Starting recorder thread")
        self.text_retrieval_thread = TextRetrievalThread()
        self.text_retrieval_thread.recorderStarted.connect(
            self.recorder_ready)
        self.text_retrieval_thread.textRetrievedLive.connect(
            self.process_live_text)
        self.text_retrieval_thread.textRetrievedFinal.connect(
            self.process_final)
        self.text_retrieval_thread.start()

        self.worker_thread = SentenceWorker(self.queue, self.tts)
        self.worker_thread.sentence_update_signal.connect(
            self.sentence_updated)
        self.worker_thread.start()

    def sentence_updated(self, full_sentences,sentence_speakers):
        self.pending_text = ""
        self.full_sentences = full_sentences
        # self.speakers = speakers
        self.sentence_speakers = sentence_speakers
        for sentence in self.full_sentences:
            sentence_text, speaker_embedding = sentence
            if sentence_text in self.pending_sentences:
                self.pending_sentences.remove(sentence_text)
        self.text_detected("")

    def start_tts(self):
        print("Loading TTS model")
        device = torch.device("cuda")
        local_models_path = os.environ.get("COQUI_MODEL_PATH")
        checkpoint = os.path.join(local_models_path, "v2.0.2")
        config = load_config((os.path.join(checkpoint, "config.json")))
        self.tts = setup_tts_model(config)
        self.tts.load_checkpoint(
            config,
            checkpoint_dir=checkpoint,
            checkpoint_path=None,
            vocab_path=None,
            eval=True,
            use_deepspeed=False,
        )
        self.tts.to(device)
        print("TTS model loaded")

    def set_text(self, text):
        self.update_thread = TextUpdateThread(text)
        self.update_thread.text_update_signal.connect(self.update_text)
        self.update_thread.start()

    def update_text(self, text):
        self.text_edit.setHtml(text)
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum())


def main():
    app = QApplication(sys.argv)

    dark_stylesheet = """
    QMainWindow {
        background-color: #323232;
    }
    QTextEdit {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
