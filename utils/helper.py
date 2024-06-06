import librosa
import numpy as np

#  Helper Functions
def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft) :
    """This functions reverts the matrix spectrograms to numpy audio"""

    def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
        """This functions reverts a spectrogram to an audio"""

        stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)

        # taking magnitude and phase of audio
        audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
        audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

        return audio_reconstruct

    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)

def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec

