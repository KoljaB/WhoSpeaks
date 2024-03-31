import os
import ffmpeg

def convert_mp3_to_wav(source_dir, target_dir, sample_rate=24000):
    """
    Converts all MP3 files in the source directory to WAV files in the target directory
    with a sample rate of 24000 Hz and mono channel.
    """
    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.mp3'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, os.path.splitext(filename)[0] + '.wav')

            try:
                # Convert MP3 to WAV using ffmpeg with specified sample rate and mono channel
                ffmpeg.input(source_file).output(target_file, ar=sample_rate, ac=1).run()
                print(f'Converted {filename} to WAV (24000 Hz, Mono)')
            except ffmpeg.Error as e:
                print(f'Error converting {filename}: {e}')

# Example usage
source_dir = 'output_sentences'  # Replace with your source directory path
target_dir = 'output_sentences_wav'  # Replace with your target directory path

convert_mp3_to_wav(source_dir, target_dir)

print('All conversions complete.')
