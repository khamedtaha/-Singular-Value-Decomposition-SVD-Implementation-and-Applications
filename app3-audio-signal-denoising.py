import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import stft, istft

# TODO: When ready, import your SVD decomposition function.
# Uncomment the line below when ready.
#from svd_lab import svd_decomposition, low_rank_svd


def read_audio(audio_path):
    """
    Reads an audio file and returns the audio data and its sample rate.
    
    Parameters:
    - audio_path: Path to the audio file.
    
    Returns:
    - audio: Audio data as a numpy array.
    - sample_rate: The sample rate of the audio.
    """
    # Read the audio file using scipy's wavfile.
    sample_rate, audio = wavfile.read(audio_path)
    
    # If the audio is stereo, convert it to mono by averaging the channels.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def short_term_fourier_transform(audio, sample_rate):
    """
    Computes the Short-Time Fourier Transform (STFT) of the audio signal.
    
    Parameters:
    - audio: Audio data as a numpy array.
    - sample_rate: The sample rate of the audio.
    
    Returns:
    - f: Array of sample frequencies.
    - t: Array of segment times.
    - Zxx: STFT of the audio signal.
    - magnitude: Magnitude spectrogram.
    - phase: Phase information of the STFT.
    """
    # Compute the STFT of the audio signal.
    f, t, Zxx = stft(audio, fs=sample_rate)
    # Separate the magnitude and phase.
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    return f, t, Zxx, magnitude, phase


def audio_denoising(sample_rate, phase, U, Sigma, V, k):
    """
    Reconstructs a denoised audio signal using a low-rank approximation of the magnitude spectrogram.
    
    Parameters:
    - sample_rate: The sample rate of the audio.
    - phase: The phase of the original STFT.
    - U: Left singular vectors from SVD of the magnitude spectrogram.
    - Sigma: Diagonal matrix of singular values from SVD.
    - V: Right singular vectors from SVD of the magnitude spectrogram.
    - k: Number of singular values to retain.
    
    Returns:
    - audio_denoised: The denoised audio signal after inverse STFT.
    """
    # TODO: Truncate the SVD components to retain only the top k singular values.
    ...
    
    # TODO: Reconstruct the magnitude spectrogram using the truncated SVD components.
    magnitude_reconstructed = ...
    
    # Recombine the reconstructed magnitude with the original phase to form the complex spectrogram.
    Zxx_reconstructed = magnitude_reconstructed * np.exp(1j * phase)
    
    # Compute the inverse STFT to obtain the denoised audio signal.
    _, audio_denoised = istft(Zxx_reconstructed, fs=sample_rate)
    
    return audio_denoised


def main():
    audios_path = r'.\application-datasets\app3-audio-signal-denoising'

    # List of audio files to test.
    audio_files = ['sp01_station_sn15.wav', 'sp07_station_sn15.wav']

    # TODO: Choose an example audio file to test.
    audio_file = audio_files[0]
    
    audio_path = os.path.join(audios_path, audio_file)
    
    # TODO: Set the number of singular values (latent factors) to retain.
    # Experiment with different values (e.g., k=5, k=15, k=25, k=50).
    k = 25

    # Read the audio file.
    audio, sample_rate = read_audio(audio_path)

    # Compute the STFT of the audio signal.
    f, t, Zxx, magnitude, phase = short_term_fourier_transform(audio, sample_rate)

    # Debugging info: print the shape of the magnitude spectrogram.
    print(magnitude.shape)

    # TODO: Apply SVD on the magnitude spectrogram using your svd_decomposition function.
    ...

    # TODO: Denoise the audio using the truncated SVD.
    audio_denoised = ...

    # Plot the original and denoised audio signals.
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(audio)
    plt.title("Original Audio Signal")
    
    plt.subplot(2, 1, 2)
    plt.plot(audio_denoised)
    plt.title(f"Denoised Audio Signal (k={k})")
    
    plt.tight_layout()
    audio_file_name = audio_file.rstrip('.wav')
    output_file = f'{audio_file_name}-k={k}-comparison.png'
    output_path = os.path.join(audios_path, output_file)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    # Normalize and save the denoised audio.
    norm_audio = audio_denoised / np.max(np.abs(audio_denoised))
    scaled_audio = np.int16(norm_audio * 32767)
    denoised_audio_file = f'{audio_file_name}-denoised-k={k}.wav'
    denoised_audio_path = os.path.join(audios_path, denoised_audio_file)
    write(denoised_audio_path, sample_rate, scaled_audio)


if __name__ == '__main__':
    main()
