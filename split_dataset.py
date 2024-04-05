import os
import re
import tempfile
from typing import List
import ffmpeg
import concurrent.futures
import time
import json
from faster_whisper import WhisperModel
import stable_whisper
import multiprocessing
from cleaner import multilingual_cleaners


# input audio file
language = "en"
input_audio_files  = [
    "input/CoinToss.mp3"
    # "input/SamuelLJackson.mp3",
    # "input/Elon Musk Podcast #49.mp3",
    # "input/Elon Musk Podcast #252.mp3",
    # "input/Elon Musk Podcast #400.mp3",
]
output_directory = 'output_sentences'

# for faster transcription disable transcript refinement
# and use a smaller model like tiny, tiny.en, small, medium
TRANSCRIPT_REFINEMENT = True
whisper_model = "tiny.en"

extend_detected_borders_start = 0.05
extend_detected_borders_end = 0.15

# https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/tokenizer.py#L597
max_text_len = 250
max_processes = 1

MB = 1024 * 1024  # Bytes in a Megabyte
CHUNK_SIZE_MB = 20  # Desired file chunk size in MB

def find_optimal_breakpoints(points: List[float], n: int) -> List[float]:
    result = []
    optimal_length = points[-1] / n
    temp = 0
    temp_a = 0
    l = len(points)
    for i in points[:l - 1]:
        if (i - temp_a) >= optimal_length:
            if optimal_length - (temp - temp_a) < (i - temp_a) - optimal_length:
                result.append(temp)
            else:
                result.append(i)
            temp_a = result[-1]
        temp = i
    return result


def split_audio_into_chunks(input_file: str, max_chunks: int,
                            silence_threshold: str = "-20dB", silence_duration: float = 2.0) -> List[str]:
    def save_chunk_to_temp_file(input_file: str, start: float, end: float) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.close()

        in_stream = ffmpeg.input(input_file)
        (
            ffmpeg.output(in_stream, temp_file.name, ss=start, t=end - start, c="copy")
            .overwrite_output()
            .run()
        )

        return temp_file.name, end - start

    def get_silence_starts(input_file: str) -> List[float]:
        silence_starts = [0.0]

        reader = (
            ffmpeg.input(input_file)
            .filter("silencedetect", n=silence_threshold, d=str(silence_duration))
            .output("pipe:", format="null")
            .run_async(pipe_stderr=True)
        )

        silence_end_re = re.compile(
            r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
        )

        while True:
            line = reader.stderr.readline().decode("utf-8")
            if not line:
                break

            match = silence_end_re.search(line)
            if match:
                silence_end = float(match.group("end"))
                silence_dur = float(match.group("dur"))
                silence_start = silence_end - silence_dur
                silence_starts.append(silence_start)

        return silence_starts

    file_extension = os.path.splitext(input_file)[1]
    metadata = ffmpeg.probe(input_file)
    duration = float(metadata["format"]["duration"])

    silence_starts = get_silence_starts(input_file)
    silence_starts.append(duration)

    temp_files = []
    lengths = []
    current_chunk_start = 0.0

    n = max_chunks
    selected_items = find_optimal_breakpoints(silence_starts, n)
    selected_items.append(duration)

    for j in range(0, len(selected_items)):
        temp_file_path, length = save_chunk_to_temp_file(input_file, current_chunk_start, selected_items[j])
        temp_files.append(temp_file_path)
        lengths.append(length)

        current_chunk_start = selected_items[j]

    return temp_files, lengths


def transcribe_file(file_name, model):
    """
    Transcribes a audio file with stable_whisper,
    returns transcript and word timestamps.
    """
    result = model.transcribe(
        file_name,
        word_timestamps=True,
        vad=True,
        language="en",
        suppress_silence=True,
        regroup=False  # disable default regrouping logic
        )

    if TRANSCRIPT_REFINEMENT:
        result = model.refine(
            file_name,
            result,
            precision=0.05,
        )

    result = (
        result.clamp_max()
        .split_by_punctuation([('.', ' '), '。', '?', '？', (',', ' '), '，'])
        .split_by_gap(.4)
        .merge_by_gap(.2, max_words=3)
        .split_by_punctuation([('.', ' '), '。', '?', '？'])
    )

    file_name_base, _ = os.path.splitext(file_name)
    result.save_as_json(file_name_base + "_transcript.json")
    return result, file_name


def format_seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:04.1f}"


# Function to calculate number of chunks based on file size
def calculate_max_chunks(file_path, chunk_size_mb):
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / MB
    return max(1, int(file_size_mb / chunk_size_mb))


def transcribe_audio(input_file: str, max_processes = 0,
                     silence_threshold: str = "-20dB", silence_duration: float = 2.0, model=None) -> str:
    if max_processes > multiprocessing.cpu_count() or max_processes == 0:
        max_processes = multiprocessing.cpu_count()


    # Calculate max chunks based on file size
    max_chunks = calculate_max_chunks(input_file, CHUNK_SIZE_MB)

    # Split the audio into chunks
    temp_files_array, lengths = split_audio_into_chunks(input_file, max_chunks, silence_threshold, silence_duration)
    print(f"Split audio into {len(temp_files_array)} chunks")
    start = time.time()
    futures = []

    # Submit each file to the thread pool and store the corresponding future object
    with concurrent.futures.ThreadPoolExecutor(max_processes) as executor:
        for file_path in temp_files_array:
            future = executor.submit(transcribe_file, file_path, model)
            futures.append(future)

    offset = 0.0
    offsets = []
    for index, file_path in enumerate(temp_files_array):
        offsets.append(offset)
        offset += lengths[index]

    sentences = []
    for future in futures:
        segments, filename = future.result()

        for segment in segments:
            if len(segment.words) == 0:
                continue

            sentence_text = ""
            sentence_start = -1
            sentence_end = -1
            for segword in segment.words:
                if sentence_start == -1:
                    sentence_start = segword.start
                sentence_text += segword.word
                sentence_end = segword.end

            file_index = temp_files_array.index(filename)
            sentence_start += offsets[file_index]
            sentence_end += offsets[file_index]

            if len(sentence_text) > max_text_len:
                print(f"Skipping long sentence: {sentence_text}")
                continue
            sentences.append((sentence_start, sentence_end, sentence_text))

    end = time.time()
    print(end - start)

    # Remember to remove the temporary files after you're done processing them
    for temp_file in temp_files_array:
        os.remove(temp_file)

    return sentences


def ends_with_sentence_ending(sentence):
    return sentence.strip().endswith(('.', '?', '!'))


# Function to merge sentences
def merge_sentences(sentences):
    merged_sentences = []
    temp_sentence = ""
    temp_start = None

    for i in range(len(sentences)):
        start, end, text = sentences[i]

        if not temp_sentence:
            temp_start = start  # Set start time for a new group of sentences

        if temp_sentence:
            text = temp_sentence + text
            temp_sentence = ""

        if not ends_with_sentence_ending(text):
            temp_sentence = text
            continue

        merged_sentences.append((temp_start, end, text.strip()))

    # Handle the last sentence if it doesn't end with a sentence-ending character
    if temp_sentence:
        last_start, last_end, _ = sentences[-1]
        merged_sentences.append((last_start, last_end, temp_sentence.strip()))

    return merged_sentences


def check_transcription_file(audio_file):
    """
    Check for an existing transcription file for the given audio file.
    Returns the path to the transcription file if it exists, None otherwise.
    """
    base, _ = os.path.splitext(audio_file)
    transcription_file = f"{base}_transcription.json"
    if os.path.exists(transcription_file):
        return transcription_file
    return None


def load_transcription(transcription_file):
    """
    Load transcription from a file.
    """
    with open(transcription_file, 'r') as file:
        return json.load(file)


def save_transcription(transcription, audio_file):
    """
    Save transcription to a file.
    """
    base, _ = os.path.splitext(audio_file)
    transcription_file = f"{base}_transcription.json"
    with open(transcription_file, 'w') as file:
        json.dump(transcription, file)


def format_seconds_to_hms_full_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    fraction_seconds = int((seconds % 1) * 10)
    seconds = int(seconds % 60)
    return f"{hours:02d}{minutes:02d}{seconds:02d}{fraction_seconds}"


def sanitize_filename(text):
    """
    Sanitize the sentence text to be safe for use in file names.
    Replace problematic characters with underscores.
    """
    return re.sub(r"[\\/*?\"<>|:']", "_", text)


def save_audio_segment(input_file, start_seconds, end_seconds, sentence_text, output_dir):
    """
    Save an audio segment from input_file between start and end times.
    The output file name is derived from start, end, and sentence_text.
    """

    # Convert seconds to formatted string
    start_formatted = format_seconds_to_hms_full_seconds(start_seconds)
    end_formatted = format_seconds_to_hms_full_seconds(end_seconds)

    # Sanitize the sentence text for file naming
    safe_sentence_text = sanitize_filename(sentence_text)
    safe_sentence_text = safe_sentence_text[:25]

    # Prepare the output file name
    file_name = f"{start_formatted}-{end_formatted}_{len(safe_sentence_text)}_{safe_sentence_text[:15].replace(' ', '_').replace('/', '_')}.mp3"

    output_path = os.path.join(output_dir, file_name)

    # Use ffmpeg library to cut the audio segment
    (
        ffmpeg
        .input(input_file, ss=start_seconds, to=end_seconds)
        .output(output_path, c="copy")
        .run(overwrite_output=True)
    )

    return output_path


def preprocess(sentences):
    for index, sentence in enumerate(sentences):
        start, end, text = sentence
        text_before = text
        text = multilingual_cleaners(text_before, language)
        if text != text_before:
            print(f"Preprocessed {text_before} to {text}")
        sentences[index] = (start, end, text)


if __name__ == "__main__":
    # Check for an existing transcription file
    model = None

    for input_audio in input_audio_files:
        print(f"Processing {input_audio}")

        transcription_file = check_transcription_file(input_audio)
        if transcription_file:
            # Load transcription from the file
            sentences = load_transcription(transcription_file)
        else:
            # Perform transcription
            if model is None:
                model = stable_whisper.load_model(whisper_model)

            sentences = transcribe_audio(
                input_audio,
                max_processes,
                silence_threshold="-20dB",
                silence_duration=2,
                model=model)

            # Merging sentences
            sentences = merge_sentences(sentences)

            # Save transcription to a file
            save_transcription(sentences, input_audio)

        # Preprocess sentences
        # Prepare texts for optimal training
        print("Preprocessing sentences texts")

        preprocess(sentences)

        # Remove sentences with 0 or negative duration before merging
        sentences = [sentence for sentence in sentences if sentence[1] > sentence[0]]

        for index, sentence in enumerate(sentences):
            start, end, text = sentence
            if end <= start:
                print(f"Pretest Skipping {text} ({start}-{end}) due to negative duration")
            if index > 0:
                _, prev_end, _ = sentences[index - 1]
                if start < prev_end:
                    print(f"Pretest Skipping {text} ({start}-{end}) due to overlap")

        # Filter out sentences with text longer than max_text_len
        final_sentences = []
        for sentence in sentences:
            if len(sentence[2]) > max_text_len:
                print(f"Removed: {sentence[2]} (Text too long: length {len(sentence[2])} > max_text_len {max_text_len})")
                continue
            final_sentences.append(sentence)

        sentences = final_sentences

        # Write sentences to disk
        print("Writing sentences to disk")
        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)


        for index, sentence in enumerate(sentences):
            start, end, text = sentence

            new_start = start
            if index > 0:
                _, prev_end, _ = sentences[index - 1]
                middle = start - (start - prev_end) / 2
                new_start = middle
                if start - middle > extend_detected_borders_start:
                    new_start = start - extend_detected_borders_start

            new_end = end
            if index < len(sentences) - 1:
                next_start, _, _ = sentences[index + 1]
                middle = end + (next_start - end) / 2
                new_end = middle
                if middle - end > extend_detected_borders_end:
                    new_end = end + extend_detected_borders_end

            startf = format_seconds_to_hms(new_start)
            endf = format_seconds_to_hms(new_end)

            print(f"{startf}-{endf}: {text}")

            if new_end < new_start:
                print(f"Skipping {text} ({new_start}-{new_end}) due to negative duration")
                continue

            save_audio_segment(input_audio, new_start, new_end, text, output_directory)

