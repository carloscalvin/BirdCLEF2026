import pytest
import numpy as np
import soundfile as sf

@pytest.fixture(scope="session")
def mock_audio_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("mock_data")
    sr = 32000
    duration = 2
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_paths = []
    for i in range(3):
        p = fn / f"test_audio_{i}.ogg"
        sf.write(str(p), y, sr)
        audio_paths.append(str(p))
    return fn, audio_paths

@pytest.fixture
def sample_audio_path(mock_audio_dir):
    _, paths = mock_audio_dir
    return paths[0]