import numpy as np
import ffmpeg
from python_speech_features import logfbank
import torch

def get_spec(
    wav, 
    from_sec, 
    duration=1, 
    sample_rate=48000, 
    aug_audio=[], 
    aud_spec_type=1, 
    use_volume_jittering=False,
    use_temporal_jittering=False,
    z_normalize=False
):
    # Temporal  jittering - get audio with 0.5 s of video clip
    if use_temporal_jittering:
        from_sec = from_sec + np.random.uniform(-0.5, 0.5)

    # Get to and from indices num seconds of audio
    fr_aud = int(np.round(from_sec * sample_rate))
    to_aud = int(np.round(from_sec * sample_rate) + sample_rate * duration)

    # Check to ensure that we never get clip longer than wav
    if fr_aud + (to_aud - fr_aud) > len(wav):
        fr_aud = len(wav) - sample_rate * duration
        to_aud = len(wav)

    # Get subset of wav
    wav = wav[fr_aud: to_aud]

    # Volume  jittering - scale volume by factor in range (0.9, 1.1)
    if use_volume_jittering:
        wav = wav * np.random.uniform(0.9, 1.1)

    # Convert to log filterbank
    if aud_spec_type == 1:
        spec = logfbank(
            wav, 
            sample_rate,
            winlen=0.02,
            winstep=0.01,
            nfilt=40,
            nfft=1024
        )
    else:
        spec = logfbank(
            wav, 
            sample_rate,
            winlen=0.02,
            winstep=0.01, #if duration==1 else 0.01,
            nfilt=257,
            nfft=1024
        )

    # Convert to 32-bit float and expand dim
    spec = spec.astype('float32')
    spec = spec.T 
    spec = np.expand_dims(spec, axis=0)
    spec = torch.as_tensor(spec)

    if z_normalize:
        spec = (spec - 1.93) / 17.89

    return spec


def load_audio(
    vid_path, from_sec, 
    duration=1, 
    sample_rate=48000, 
    aug_audio=[], 
    aud_spec_type=1, 
    use_volume_jittering=False,
    use_temporal_jittering=False,
    z_normalize=False,
):

    # Load wav file @ sample_rate
    out, _ = (
        ffmpeg
        .input(vid_path)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=sample_rate)
        .run(quiet=True)
    )
    wav = (
        np
        .frombuffer(out, np.int16)
    )
    # Get spectrogram
    spec = get_spec(
        wav, 
        from_sec, 
        duration=duration, 
        sample_rate=sample_rate, 
        aug_audio=aug_audio, 
        aud_spec_type=aud_spec_type, 
        use_volume_jittering=use_volume_jittering, 
        use_temporal_jittering=use_temporal_jittering,
        z_normalize=z_normalize,
    )
    return spec



if __name__ == '__main__':

    vid_path = '../data/train/class_1/test_video.mp4'
    from_sec = 0
    aud_spec_type = 0

    spectogram = load_audio(
                    vid_path, from_sec, 
                    duration=1, 
                    sample_rate=24000, 
                    aug_audio=[], 
                    aud_spec_type=aud_spec_type, 
                    use_volume_jittering=False,
                    use_temporal_jittering=False,
                    z_normalize=False,
                )
    
    pass