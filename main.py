import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def spectral_subtraction(input_file, output_file, noise_level=1.5):
    rate, data = wav.read(input_file)
    spectrum = np.fft.fft(data)
    magnitude_spectrum = np.abs(spectrum)
    noise_threshold = noise_level * np.median(magnitude_spectrum)
    cleaned_spectrum = np.maximum(0, magnitude_spectrum - noise_threshold)
    cleaned_data = np.fft.ifft(cleaned_spectrum).real.astype(np.int16)
    wav.write(output_file, rate, cleaned_data)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Original Spectrum')
    plt.plot(magnitude_spectrum)

    plt.subplot(2, 1, 2)
    plt.title('Cleaned Spectrum')
    plt.plot(cleaned_spectrum)

    plt.show()

# Example usage
input_file = "input_voice_recording.wav"
output_file = "output_cleaned_voice.wav"
spectral_subtraction(input_file, output_file)
