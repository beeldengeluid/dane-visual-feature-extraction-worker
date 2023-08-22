import tensorflow as tf


def make_spectrogram(audio,
                     stft_length=2048,
                     stft_step=1024,
                     stft_pad_end=True,
                     use_mel=True,
                     mel_lower_edge_hertz=80.,
                     mel_upper_edge_hertz=7600.,
                     mel_sample_rate=48000.,
                     mel_num_bins=40,
                     use_log=True,
                     log_eps=1.,
                     log_scale=10000.):
    """Computes (mel) spectrograms for signals t."""
    stfts = tf.signal.stft(audio,
                         frame_length=stft_length,
                         frame_step=stft_step,
                         fft_length=stft_length,
                         pad_end=stft_pad_end)
    spectrogram = tf.abs(stfts)
    if use_mel:
        num_spectrogram_bins = spectrogram.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                mel_num_bins, num_spectrogram_bins, mel_sample_rate,
                mel_lower_edge_hertz, mel_upper_edge_hertz)
        spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        spectrogram.set_shape(spectrogram.shape[:-1] +
                          linear_to_mel_weight_matrix.shape[-1:])

    if use_log:
        spectrogram = tf.log(log_eps + log_scale * spectrogram)
    return spectrogram



def raw_audio_to_spectrogram(raw_audio, # some waveform of shape 1 x n_samples
                             sample_rate=48000,
                             stft_length=0.032,
                             stft_step=0.016,
                             mel_bins=80,
                             rm_audio=False):
    """Computes audio spectrogram and eventually removes raw audio."""
    stft_length = int(sample_rate * stft_length)
    stft_step = int(sample_rate * stft_step)
    mel_spectrogram = make_spectrogram(audio=raw_audio,
                         mel_sample_rate=sample_rate,
                         stft_length=stft_length,
                         stft_step=stft_step,
                         mel_num_bins=mel_bins,
                         use_mel=True)
    return mel_spectrogram