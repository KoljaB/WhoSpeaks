from yt_dlp import YoutubeDL
from os.path import exists, join, splitext
import os

urls = [
    "https://www.youtube.com/watch?v=JN3KPFbWCy8", # Elon Musk / Lex Fridman Round 4
    "https://www.youtube.com/watch?v=DxREm3s1scA", # Elon Musk / Lex Fridman Round 3
    "https://www.youtube.com/watch?v=smK9dgdTl40", # Elon Musk / Lex Fridman Round 2
    "https://www.youtube.com/watch?v=dEv99vxKjVI", # Elon Musk / Lex Fridman Round 1
]
directory = "input"


def fetch_youtube(
    url: str,
    filetype: str,
    directory: str = "downloaded_files"
):

    """
    Downloads a specific type of file (video, audio, or muted video)
    from the provided YouTube URL.

    Args:
        url (str): The URL of the YouTube video to be downloaded.
        filetype (str): Type of file to download - 'video', 'audio',
            or 'muted_video'.
        directory (str): The directory to download the file to.

    Returns:
        str: The filename of the downloaded file.
    """
    if directory and not exists(directory):
        os.makedirs(directory)

    if filetype == 'video':
        # Download video with audio
        outtmpl = join(directory, '%(title)s.%(ext)s')
        ydl_opts = {
            'format': 'best',
            'outtmpl': outtmpl,
            'noplaylist': True,
        }
    elif filetype == 'audio':
        # Download audio only
        outtmpl = join(directory, '%(title)s_audio.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'noplaylist': True,
        }
    elif filetype == 'mp3_audio':
        # Download audio as MP3
        outtmpl = join(directory, '%(title)s.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'noplaylist': True,
        }   
    elif filetype == 'muted_video':
        # Download video without audio
        outtmpl = join(directory, '%(title)s_mutedvideo.%(ext)s')
        ydl_opts = {
            'format': 'bestvideo',
            'outtmpl': outtmpl,
            'noplaylist': True,
        }
    else:
        raise ValueError(
            "Invalid filetype. Choose 'video', 'audio', or 'muted_video'."
        )

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

    return downloaded_file

for url in urls:
    audio_file = fetch_youtube(url, 'mp3_audio', directory)
