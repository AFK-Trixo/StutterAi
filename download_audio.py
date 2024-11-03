import os
import pathlib
import subprocess
import numpy as np
import argparse
import requests

parser = argparse.ArgumentParser(description='Download raw audio files for SEP-28k or FluencyBank and convert to 16k hz mono wavs.')
parser.add_argument('--episodes', type=str, required=True,
                    help='Path to the labels csv files (e.g., SEP-28k_episodes.csv)')
parser.add_argument('--wavs', type=str, default="wavs",
                    help='Path where audio files from download_audio.py are saved')

args = parser.parse_args()
episode_uri = args.episodes
wav_dir = args.wavs

# Load episode data
table = np.loadtxt(episode_uri, dtype=str, delimiter=",")
urls = table[:, 2]
n_items = len(urls)

audio_types = [".mp3", ".m4a", ".mp4"]

for i in range(n_items):
    # Get show/episode IDs and clean up any extra spaces
    show_abrev = table[i, -2].strip().replace(" ", "_")
    ep_idx = table[i, -1].strip().replace(" ", "_")
    episode_url = table[i, 2].strip()

    # Check file extension
    ext = ''
    for audio_ext in audio_types:
        if audio_ext in episode_url:
            ext = audio_ext
            break

    # Ensure the base folder exists for this episode
    episode_dir = pathlib.Path(f"{wav_dir}/{show_abrev}/")
    os.makedirs(episode_dir, exist_ok=True)

    # Get file paths
    audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
    wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")

    # Check if this file has already been downloaded
    if os.path.exists(wav_path):
        continue

    print("Processing", show_abrev, ep_idx)

    # Download raw audio file
    if not os.path.exists(audio_path_orig):
        try:
            print(f"Downloading {episode_url} to {audio_path_orig}")
            response = requests.get(episode_url, stream=True)
            response.raise_for_status()  # Check if the request was successful
            with open(audio_path_orig, "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=8192):
                    audio_file.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {episode_url}: {e}")
            continue

    # Convert to 16khz mono wav file
    if os.path.exists(audio_path_orig):
        line = f"ffmpeg -i {audio_path_orig} -ac 1 -ar 16000 {wav_path}"
        process = subprocess.Popen(line, shell=True)
        process.wait()

        # Remove the original mp3/m4a file after conversion
        os.remove(audio_path_orig)
