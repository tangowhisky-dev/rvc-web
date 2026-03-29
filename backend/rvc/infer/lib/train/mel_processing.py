import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import logging

logger = logging.getLogger(__name__)

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


# Reusable banks
mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Args:
        y             :: (B, T) - Audio waveforms
        n_fft
        sampling_rate
        hop_size
        win_size
        center
    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """

    # Window - Cache if needed
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # MelBasis - Cache if needed
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Convert waveform into Mel-frequency Log-amplitude spectrogram.

    Args:
        y       :: (B, T)           - Waveforms
    Returns:
        melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """
    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return melspec


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """Multi-scale mel spectrogram loss for RefineGAN training.

    Computes L1 loss across 7 STFT scales, operating on log-mel spectrograms.
    Ported from Applio/rvc/train/mel_processing.py.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: list = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: list = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn or torch.nn.L1Loss()
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params = list(zip(n_mels, window_lengths))
        self.hann_window: dict = {}
        self.mel_banks: dict = {}

    def mel_spectrogram(self, wav: torch.Tensor, n_mels: int, window_length: int):
        dtype_device = f"{wav.dtype}_{wav.device}"
        win_key = f"{window_length}_{dtype_device}"
        mel_key = f"{n_mels}_{dtype_device}"

        if win_key not in self.hann_window:
            self.hann_window[win_key] = torch.hann_window(
                window_length, device=wav.device, dtype=torch.float32
            )

        wav_sq = wav.squeeze(1).float()
        stft = torch.stft(
            wav_sq,
            n_fft=window_length,
            hop_length=window_length // 4,
            window=self.hann_window[win_key],
            return_complex=True,
        )
        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

        if mel_key not in self.mel_banks:
            self.mel_banks[mel_key] = torch.from_numpy(
                librosa_mel_fn(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None,
                )
            ).to(device=wav.device, dtype=torch.float32)

        return torch.matmul(self.mel_banks[mel_key], magnitude)

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_mels, win in self.stft_params:
            real_mels = self.mel_spectrogram(real, n_mels, win)
            fake_mels = self.mel_spectrogram(fake, n_mels, win)
            real_log = torch.log(real_mels.clamp(min=1e-5)) / self.log_base
            fake_log = torch.log(fake_mels.clamp(min=1e-5)) / self.log_base
            loss = loss + self.loss_fn(real_log, fake_log)
        return loss
