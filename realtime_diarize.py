from PyQt6.QtWidgets import QApplication, QTextEdit, QMainWindow, QLabel, QVBoxLayout, QWidget, QDoubleSpinBox, QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QEvent, QTimer
from sklearn.cluster import AgglomerativeClustering, KMeans
from TTS.tts.models import setup_model as setup_tts_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from RealtimeSTT import AudioToTextRecorder
from TTS.config import load_config
import numpy as np
import pyaudio
import queue
import torch
import wave
import sys
import os

SILENCE_THRESHS = [0, 0.4]
FINAL_TRANSCRIPTION_MODEL = "large-v2"
FINAL_BEAM_SIZE = 5
REALTIME_TRANSCRIPTION_MODEL = "distil-small.en"
REALTIME_BEAM_SIZE = 5
TRANSCRIPTION_LANGUAGE = "en"
SILERO_SENSITIVITY = 0.4
WEBRTC_SENSITIVITY = 3
MIN_LENGTH_OF_RECORDING = 0.7
PRE_RECORDING_BUFFER_DURATION = 0.35
INIT_TWO_SPEAKER_THRESHOLD = 17
INIT_SILHOUETTE_DIFF_THRESHOLD = 0.0001

FAST_SENTENCE_END = True
USE_MICROPHONE = True
LOOPBACK_DEVICE_NAME = "stereomix"
LOOPBACK_DEVICE_HOST_API = 0

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

two_speaker_threshold = INIT_TWO_SPEAKER_THRESHOLD
silhouette_diff_threshold = INIT_SILHOUETTE_DIFF_THRESHOLD


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
            'model': FINAL_TRANSCRIPTION_MODEL,
            'language': TRANSCRIPTION_LANGUAGE,
            'silero_sensitivity': SILERO_SENSITIVITY,
            'webrtc_sensitivity': WEBRTC_SENSITIVITY,
            'post_speech_silence_duration': SILENCE_THRESHS[1],
            'min_length_of_recording': MIN_LENGTH_OF_RECORDING,
            'pre_recording_buffer_duration': PRE_RECORDING_BUFFER_DURATION,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': REALTIME_TRANSCRIPTION_MODEL,
            'on_realtime_transcription_update': self.live_text_detected,
            'beam_size': FINAL_BEAM_SIZE,
            'beam_size_realtime': REALTIME_BEAM_SIZE,
            'buffer_size': BUFFER_SIZE,
            'sample_rate': SAMPLE_RATE,
        }

        self.recorder = AudioToTextRecorder(**recorder_config)
        self.recorderStarted.emit()

        def process_text(text):
            bytes = self.recorder.last_transcription_bytes
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

        # Selecting the input device based on USE_MICROPHONE flag
        if USE_MICROPHONE:
            input_device_index = 0  # Default input device (microphone)
        else:
            input_device_index, _ = find_stereo_mix_index()
            if input_device_index is None:
                print("Loopback / Stereo Mix device not found")
                print("Available devices:\n", devices)
                self.audio.terminate()
                exit()
            else:
                print(f"Stereo Mix device found at index: {input_device_index}")

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE,
            input_device_index=input_device_index)
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

    # Safety check using KMeans for initial speaker detection
    def determine_optimal_cluster_count(self, embeddings_scaled):
        num_embeddings = len(embeddings_scaled)
        if num_embeddings <= 1:
            # Only one embedding, so only one speaker
            return 1
        
        # Determine single or multiple speakers
        # K-means Clustering with k=2
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
        distances = kmeans.transform(embeddings_scaled)
        avg_distance = np.mean(np.min(distances, axis=1))
        distance_threshold = two_speaker_threshold  # Threshold to decide if we have one or multiple speakers

        # Check if the average distance is below threshold for single speaker
        if avg_distance < distance_threshold:
            print(f"Single Speaker: low embedding distance: {avg_distance} < {distance_threshold}.")
            return 1

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


        # Find the optimal number of clusters
        # It's the point before the silhouette score starts to decrease significantly
        optimal_cluster_count = 2
        for i in range(1, len(silhouette_scores)):
            if silhouette_scores[i] < silhouette_scores[i-1] + silhouette_diff_threshold:
                optimal_cluster_count = range_clusters[i-1]
                break

        return optimal_cluster_count

    def process_speakers(self):
        embeddings = [speaker_embedding for _, speaker_embedding in self.full_sentences]

        # Standard scaling
        embeddings_array = np.array(embeddings)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)

        optimal_cluster_count = self.determine_optimal_cluster_count(embeddings_scaled)

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

        # Create the main layout as horizontal
        self.mainLayout = QHBoxLayout()

        # Add the text edit to the main layout
        self.text_edit = QTextEdit(self)
        self.mainLayout.addWidget(self.text_edit, 1)

        # Create the right layout for controls and add them to the main layout
        self.rightLayout = QVBoxLayout()
        self.rightLayout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align controls to the top
        self.create_controls()

        # Create a container for the right layout
        self.rightContainer = QWidget()
        self.rightContainer.setLayout(self.rightLayout)
        self.mainLayout.addWidget(self.rightContainer, 0)  # Controls get the space they need

        # Set the main layout to the central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        self.setStyleSheet("""
            QLabel {
                color: #ddd;
            }
            QDoubleSpinBox {
                background: #333;
                color: #ddd;
                border: 1px solid #555;
                margin-bottom: 22px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Arial';
                font-size: 16pt;
            }
        """)

    def create_controls(self):

        self.two_speaker_threshold_desc = QLabel("For one or two speakers differentiation:")
        self.two_speaker_threshold_label = QLabel("Two cluster similarity (0.1-100)")
        self.two_speaker_threshold_spinbox = QDoubleSpinBox()
        self.two_speaker_threshold_spinbox.setRange(0.1, 100)
        self.two_speaker_threshold_spinbox.setSingleStep(0.1)
        self.two_speaker_threshold_spinbox.setValue(two_speaker_threshold)
        self.two_speaker_threshold_spinbox.valueChanged.connect(self.update_two_speaker_threshold)

        self.silhouette_diff_threshold_desc = QLabel("For more than two speakers differentiation:")
        self.silhouette_diff_threshold_label = QLabel("Silhouette similarity (0.001-1)")
        self.silhouette_diff_threshold_spinbox = QDoubleSpinBox()
        self.silhouette_diff_threshold_spinbox.setDecimals(5)
        self.silhouette_diff_threshold_spinbox.setRange(0, 0.01)
        self.silhouette_diff_threshold_spinbox.setSingleStep(0.00001)
        self.silhouette_diff_threshold_spinbox.setValue(silhouette_diff_threshold)
        self.silhouette_diff_threshold_spinbox.valueChanged.connect(self.update_silhouette_diff_threshold)

        self.two_speaker_threshold_label.setToolTip(
            "Adjust this threshold to control how the program differentiates between one or two speakers. "
            "Lower values mean only highly distinct voices are considered separate speakers. "
            "Higher values allow more leniency in identifying different speakers."
        )

        self.silhouette_diff_threshold_spinbox.setToolTip(
            "This value determines the required increase in similarity score to identify an additional speaker. "
            "Lower values make it easier to identify more speakers. "
            "Higher values prevent too many speakers from being identified, especially in noisy conditions."
        )

        # Add the controls to the right layout
        self.rightLayout.addWidget(self.two_speaker_threshold_desc)
        self.rightLayout.addWidget(self.two_speaker_threshold_label)
        self.rightLayout.addWidget(self.two_speaker_threshold_spinbox)

        self.rightLayout.addWidget(self.silhouette_diff_threshold_desc)
        self.rightLayout.addWidget(self.silhouette_diff_threshold_label)
        self.rightLayout.addWidget(self.silhouette_diff_threshold_spinbox)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_state)
        self.rightLayout.addWidget(self.clear_button)

    def clear_state(self):
        # Clear text edit
        self.text_edit.clear()

        # Reset state variables
        self.displayed_text = ""
        self.last_realtime_text = ""
        self.full_sentences = []
        self.sentence_speakers = []
        self.pending_sentences = []
        self.worker_thread.full_sentences = []
        self.worker_thread.sentence_speakers = []
        self.worker_thread.speakers = []

        # Optional: Provide a message in text edit to indicate clearing
        self.text_edit.setHtml("<i>All cleared. Ready for new input.</i>")

    def update_ui(self):
        self.worker_thread.process_speakers()
        self.sentence_updated(
            self.worker_thread.full_sentences,
            self.worker_thread.sentence_speakers)

    def update_two_speaker_threshold(self, value):
        global two_speaker_threshold
        two_speaker_threshold = value
        self.update_ui()

    def update_silhouette_diff_threshold(self, value):
        global silhouette_diff_threshold
        silhouette_diff_threshold = value
        self.update_ui()

    def showEvent(self, event):
        super().showEvent(event)
        if event.type() == QEvent.Type.Show:
            if not self.initialized:
                self.initialized = True
                self.resize(1200, 800)
                self.update_text("<i>Please wait until app is loaded</i>")

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
        self.update_text("<i>Ready to record</i>")

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
