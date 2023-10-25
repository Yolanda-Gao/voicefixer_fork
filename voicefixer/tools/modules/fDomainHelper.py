# from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from voicefixer.tools.modules.pqmf import PQMF
from typing import Any, Dict, Union

class DFTBase(nn.Module):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix.
        """
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        r"""PyTorch implementation of STFT with Conv1d. The function has the 
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(data=fft_window, size=n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
            groups=1, bias=False)

        # Initialize Conv1d weights.
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate STFT of batch of signals.

        Args: 
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, 1, time_steps, n_fft // 2 + 1)
            imag: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    r"""Calculate magnitude and phase from real and imag part of signals.

    Args:
        real: tensor, real part of signals
        imag: tensor, imag part of signals

    Returns:
        mag: tensor, magnitude of signals
        cos: tensor, cosine of phases of signals
        sin: tensor, sine of phases of signals
    """
    mag = (real ** 2 + imag ** 2) ** 0.5
    cos = real / torch.clamp(mag, 1e-10, np.inf)
    sin = imag / torch.clamp(mag, 1e-10, np.inf)

    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True, 
        onnx=False, frames_num=None, device=None):
        """PyTorch implementation of ISTFT with Conv1d. The function has the 
        same output as librosa.istft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
            onnx: bool, set to True when exporting trained model to ONNX. This
                will replace several operations to operators supported by ONNX.
            frames_num: None | int, number of frames of audio clips to be 
                inferneced. Only useable when onnx=True.
            device: None | str, device of ONNX. Only useable when onnx=True.
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        if not onnx:
            assert frames_num is None, "When onnx=False, frames_num must be None!"
            assert device is None, "When onnx=False, device must be None!"

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.onnx = onnx

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # Initialize Conv1d modules for calculating real and imag part of DFT.
        self.init_real_imag_conv()

        # Initialize overlap add window for reconstruct time domain signals.
        self.init_overlap_add_window()

        if self.onnx:
            # Initialize ONNX modules.
            self.init_onnx_modules(frames_num, device)
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_real_imag_conv(self):
        r"""Initialize Conv1d for calculating real and imag part of DFT.
        """
        self.W = self.idft_matrix(self.n_fft) / self.n_fft

        self.conv_real = nn.Conv1d(in_channels=self.n_fft, out_channels=self.n_fft,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=self.n_fft, out_channels=self.n_fft,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False)

        ifft_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        # (win_length,)

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(data=ifft_window, size=self.n_fft)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

    def init_overlap_add_window(self):
        r"""Initialize overlap add window for reconstruct time domain signals.
        """
        
        ola_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        # (win_length,)

        ola_window = librosa.util.normalize(ola_window, norm=None) ** 2
        ola_window = librosa.util.pad_center(data=ola_window, size=self.n_fft)
        ola_window = torch.Tensor(ola_window)

        self.register_buffer('ola_window', ola_window)
        # (win_length,)

    def init_onnx_modules(self, frames_num, device):
        r"""Initialize ONNX modules.

        Args:
            frames_num: int
            device: str | None
        """

        # Use Conv1d to implement torch.flip(), because torch.flip() is not 
        # supported by ONNX.
        self.reverse = nn.Conv1d(in_channels=self.n_fft // 2 + 1,
            out_channels=self.n_fft // 2 - 1, kernel_size=1, bias=False)

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1 : -1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        # (n_fft // 2 - 1, n_fft // 2 + 1, 1)

        # Use nn.ConvTranspose2d to implement torch.nn.functional.fold(), 
        # because torch.nn.functional.fold() is not supported by ONNX.
        self.overlap_add = nn.ConvTranspose2d(in_channels=self.n_fft,
            out_channels=1, kernel_size=(self.n_fft, 1), stride=(self.hop_length, 1), bias=False)

        self.overlap_add.weight.data = torch.Tensor(np.eye(self.n_fft)[:, None, :, None])
        # (n_fft, 1, n_fft, 1)

        if frames_num:
            # Pre-calculate overlap-add window sum for reconstructing signals
            # when using ONNX.
            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
        else:
            self.ifft_window_sum = []

    def forward(self, real_stft, imag_stft, length):
        r"""Calculate inverse STFT.

        Args:
            real_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            imag_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            length: int
        
        Returns:
            real: (batch_size, data_length), output signals.
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        batch_size, _, frames_num, _ = real_stft.shape

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Get full stft representation from spectrum using symmetry attribute.
        if self.onnx:
            full_real_stft, full_imag_stft = self._get_full_stft_onnx(real_stft, imag_stft)
        else:
            full_real_stft, full_imag_stft = self._get_full_stft(real_stft, imag_stft)
        # full_real_stft: (batch_size, n_fft, time_steps)
        # full_imag_stft: (batch_size, n_fft, time_steps)

        # Calculate IDFT frame by frame.
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        # (batch_size, n_fft, time_steps)

        # Overlap add signals in frames to reconstruct signals.
        if self.onnx:
            y = self._overlap_add_divide_window_sum_onnx(s_real, frames_num)
        else:
            y = self._overlap_add_divide_window_sum(s_real, frames_num)
        # y: (batch_size, audio_samples + win_length,)
        
        y = self._trim_edges(y, length)
        # (batch_size, audio_samples,)
            
        return y

    def _get_full_stft(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """
        full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)

        return full_real_stft, full_imag_stft

    def _get_full_stft_onnx(self, real_stft, imag_stft):
        r"""Get full stft representation from spectrum using symmetry attribute
        for ONNX. Replace several pytorch operations in self._get_full_stft() 
        that are not supported by ONNX.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """

        # Implement torch.flip() with Conv1d.
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, - self.reverse(imag_stft)), dim=1)

        return full_real_stft, full_imag_stft

    def _overlap_add_divide_window_sum(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """
        
        output_samples = (s_real.shape[-1] - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        # Overlap-add signals in frames to signals. Ref: 
        # asteroid_filterbanks.torch_stft_fb.torch_stft_fb() from
        # https://github.com/asteroid-team/asteroid-filterbanks
        y = torch.nn.functional.fold(input=s_real, output_size=(1, output_samples), 
            kernel_size=(1, self.win_length), stride=(1, self.hop_length))
        # (batch_size, 1, 1, audio_samples,)
        
        y = y[:, 0, 0, :]
        # (batch_size, audio_samples)

        # Get overlap-add window sum to be divided.
        ifft_window_sum = self._get_ifft_window(frames_num)
        # (audio_samples,)

        # Following code is abandaned for divide overlap-add window, because
        # not supported by half precision training and ONNX.
        # min_mask = ifft_window_sum.abs() < 1e-11
        # y[:, ~min_mask] = y[:, ~min_mask] / ifft_window_sum[None, ~min_mask]
        # # (batch_size, audio_samples)

        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)

        return y

    def _get_ifft_window(self, frames_num):
        r"""Get overlap-add window sum to be divided.

        Args:
            frames_num: int

        Returns:
            ifft_window_sum: (audio_samlpes,), overlap-add window sum to be 
            divided.
        """
        
        output_samples = (frames_num - 1) * self.hop_length + self.win_length
        # (audio_samples,)

        window_matrix = self.ola_window[None, :, None].repeat(1, 1, frames_num)
        # (batch_size, win_length, time_steps)

        ifft_window_sum = F.fold(input=window_matrix, 
            output_size=(1, output_samples), kernel_size=(1, self.win_length), 
            stride=(1, self.hop_length))
        # (1, 1, 1, audio_samples)
        
        ifft_window_sum = ifft_window_sum.squeeze()
        # (audio_samlpes,)

        return ifft_window_sum

    def _overlap_add_divide_window_sum_onnx(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals for ONNX. 
        Replace several pytorch operations in 
        self._overlap_add_divide_window_sum() that are not supported by ONNX.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """

        s_real = s_real[..., None]
        # (batch_size, n_fft, time_steps, 1)

        # Implement overlap-add with Conv1d, because torch.nn.functional.fold()
        # is not supported by ONNX.
        y = self.overlap_add(s_real)[:, 0, :, 0]    
        # y: (batch_size, samples_num)
        
        if len(self.ifft_window_sum) != y.shape[1]:
            device = s_real.device

            self.ifft_window_sum = self._get_ifft_window_sum_onnx(frames_num, device)
            # (audio_samples,)

        # Use torch.clamp() to prevent from underflow to make sure all 
        # operations are supported by ONNX.
        ifft_window_sum = torch.clamp(self.ifft_window_sum, 1e-11, np.inf)
        # (audio_samples,)

        y = y / ifft_window_sum[None, :]
        # (batch_size, audio_samples,)
        
        return y

    def _get_ifft_window_sum_onnx(self, frames_num, device):
        r"""Pre-calculate overlap-add window sum for reconstructing signals when
        using ONNX.

        Args:
            frames_num: int
            device: str | None

        Returns:
            ifft_window_sum: (audio_samples,)
        """
        
        ifft_window_sum = librosa.filters.window_sumsquare(window=self.window, 
            n_frames=frames_num, win_length=self.win_length, n_fft=self.n_fft, 
            hop_length=self.hop_length)
        # (audio_samples,)

        ifft_window_sum = torch.Tensor(ifft_window_sum)

        if device:
            ifft_window_sum = ifft_window_sum.to(device)

        return ifft_window_sum

    def _trim_edges(self, y, length):
        r"""Trim audio.

        Args:
            y: (audio_samples,)
            length: int

        Returns:
            (trimmed_audio_samples,)
        """
        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]

        return y


class FDomainHelper(nn.Module):

    def __init__(
        self,
        window_size=2048,
        hop_size=441,
        center=True,
        pad_mode="reflect",
        window="hann",
        freeze_parameters=True,
        subband=None,
        root="/Users/admin/Documents/projects/",
    ):
        super(FDomainHelper, self).__init__()
        self.subband = subband
        # assert torchlibrosa.__version__ == "0.0.7", "Error: Found torchlibrosa version %s. Please install 0.0.7 version of torchlibrosa by: pip install torchlibrosa==0.0.7." % torchlibrosa.__version__
       #  if self.subband is None:
       #      self.n_fft=window_size
       #      self.hop_length=hop_size
       #      self.win_length=window_size

       #  else:
       #      self.n_fft=window_size // self.subband
       #      self.hop_length=hop_size // self.subband
       #      self.win_length=window_size // self.subband

       #  self.window=window,
       #  self.center=center,
       #  self.pad_mode=pad_mode,


        if subband is not None and root is not None:
            self.qmf = PQMF(subband, 64, root)

        if self.subband is None:
            self.stft = STFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = ISTFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )
        else:
            self.stft = STFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = ISTFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

   # def STFT(self, waveforms) -> Tensor:
   #   """Returns complex-valued spectrogram in (n_fft/2 + 1) dim."""
   #   if not torch.is_tensor(waveforms):
   #     waveforms = torch.from_numpy(waveforms)
   #   # [..., n_fft/2 + 1, n_frames]
   #   stft = torch.stft(
   #       waveforms,
   #       n_fft=self.n_fft,
   #       hop_length=self.hop_length,
   #       win_length=self.win_length,
   #       center=self.center,
   #       pad_mode=self.pad_mode,
   #       window=self.window.to(waveforms.device),
   #       return_complex=True)
   #   return stft
  
   # def ISTFT(self, spectrogram: Union[Tensor, np.ndarray], length) -> Tensor:
   #   """Returns to wav from spectrogram."""
   #   istft = torch.istft(
   #       spectrogram,
   #       n_fft=self.n_fft,
   #       hop_length=self.hop_length,
   #       win_length=self.win_length,
   #       center=self.center,
   #       window=self.window,
   #       length=length)
   #   return istft

   # def magphase(real, imag):
   #   r"""Calculate magnitude and phase from real and imag part of signals.
  
   #   Args:
   #       real: tensor, real part of signals
   #       imag: tensor, imag part of signals
  
   #   Returns:
   #       mag: tensor, magnitude of signals
   #       cos: tensor, cosine of phases of signals
   #       sin: tensor, sine of phases of signals
   #   """
   #   mag = (real ** 2 + imag ** 2) ** 0.5
   #   cos = real / torch.clamp(mag, 1e-10, np.inf)
   #   sin = imag / torch.clamp(mag, 1e-10, np.inf)
  
   #   return mag, cos, sin

    def complex_spectrogram(self, input, eps=0.0):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def reverse_complex_spectrogram(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]
        wav = self.istft(input[:, 0:1, ...], input[:, 1:2, ...], length=length)
        return wav

    def spectrogram(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        return torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def spectrogram_phase_to_wav(self, sps, coss, sins, length):
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(
                self.istft(
                    sps[:, i : i + 1, ...] * coss[:, i : i + 1, ...],
                    sps[:, i : i + 1, ...] * sins[:, i : i + 1, ...],
                    length,
                )
            )
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res, dim=1)

    def wav_to_spectrogram(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size,channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, channel, :], eps=eps))
        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(
                self.istft(
                    spectrogram[:, channel : channel + 1, :, :] * cos,
                    spectrogram[:, channel : channel + 1, :, :] * sin,
                    length,
                )
            )

        output = torch.stack(wav_list, dim=1)
        return output

    # todo the following code is not bug free!
    def wav_to_complex_spectrogram(self, input, eps=0.0):
        # [batchsize , channels, samples]
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self.complex_spectrogram(input[:, channel, :], eps=eps))
        return torch.cat(res, dim=1)

    def complex_spectrogram_to_wav(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        # return  [batchsize, channels, samples]
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(
                self.reverse_complex_spectrogram(
                    input[:, 2 * i : 2 * i + 2, ...], eps=eps, length=length
                )
            )
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs, dim=1)

    def wav_to_complex_subband_spectrogram(self, input, eps=0.0):
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        subspec = self.wav_to_complex_spectrogram(subwav)
        return subspec

    def complex_subband_spectrogram_to_wav(self, input, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.complex_spectrogram_to_wav(input)
        data = self.qmf.synthesis(subwav)
        return data

    def wav_to_mag_phase_subband_spectrogram(self, input, eps=1e-8):
        """
        :param input:
        :param eps:
        :return:
            loss = torch.nn.L1Loss()
            models = FDomainHelper(subband=4)
            data = torch.randn((3,1, 44100*3))

            sps, coss, sins = models.wav_to_mag_phase_subband_spectrogram(data)
            wav = models.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

            print(loss(data,wav))
            print(torch.max(torch.abs(data-wav)))

        """
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        sps, coss, sins = self.wav_to_spectrogram_phase(subwav, eps=eps)
        return sps, coss, sins

    def mag_phase_subband_spectrogram_to_wav(self, sps, coss, sins, length, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.spectrogram_phase_to_wav(
            sps, coss, sins, length + self.qmf.pad_samples // self.qmf.N
        )
        data = self.qmf.synthesis(subwav)
        return data
