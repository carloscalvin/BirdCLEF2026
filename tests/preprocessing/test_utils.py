import numpy as np
import pytest
from src.preprocessing import utils

def test_calculate_rms():
    y = np.ones(100)
    rms = utils.calculate_rms(y)
    assert rms == pytest.approx(1.0, abs=1e-5)
    y_silence = np.zeros(100)
    assert utils.calculate_rms(y_silence) == pytest.approx(1e-8, abs=1e-9)

def test_get_duration(sample_audio_path):
    duration = utils.get_duration(sample_audio_path)
    assert duration == pytest.approx(2.0, abs=0.1)

def test_resample_shape(sample_audio_path):
    target_sr = 16000
    y, sr_orig = utils.resample(sample_audio_path, target_sr)
    assert sr_orig == 32000
    expected_len = 2 * target_sr
    assert len(y) == expected_len

def test_mix_multiple_audios_logic():
    sr = 32000
    duration = 5
    noise = np.random.rand(sr * duration).astype(np.float32)
    bird = np.random.rand(sr * duration).astype(np.float32)
    mixed = utils.mix_multiple_audios(
        bird_chunks=[bird],
        noise_chunk=noise,
        min_snr_db=100,
        max_snr_db=100,
        duration=duration,
        sr=sr
    )
    assert mixed.shape == (sr * duration,)
    assert np.max(np.abs(mixed)) > 0
    assert np.max(np.abs(mixed)) <= 1.0