import os
import re
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# text normalization
_RU_ALLOWED = re.compile(r"[^а-яё0-9\s-]+")

def normalize_ru_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ё", "е")
    s = s.replace("-", " ")
    s = _RU_ALLOWED.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


UNITS = {
    "ноль": 0,
    "один": 1, "одна": 1, "одно": 1,
    "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
}
TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
THOUSANDS = {"тысяча", "тысячи", "тысяч"}


def parse_0_999(tokens: List[str]) -> int:
    total = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in HUNDREDS:
            total += HUNDREDS[t]
        elif t in TEENS:
            total += TEENS[t]
        elif t in TENS:
            total += TENS[t]
            # optional unit after tens
            if i + 1 < len(tokens) and tokens[i+1] in UNITS:
                total += UNITS[tokens[i+1]]
                i += 1
        elif t in UNITS:
            total += UNITS[t]
        # else: unknown token -> skip
        i += 1
    return total


def ru_words_to_int(text: str) -> int:
    text = normalize_ru_text(text)
    if not text:
        raise ValueError("empty text")

    tokens = text.split()

    th_pos = None
    for idx, tok in enumerate(tokens):
        if tok in THOUSANDS:
            th_pos = idx
            break

    if th_pos is None:
        return parse_0_999(tokens)

    left = tokens[:th_pos]
    right = tokens[th_pos+1:]

    mult = parse_0_999(left) if left else 1
    if mult == 0:
        mult = 1

    value = mult * 1000 + parse_0_999(right)
    return value

# --------------------------------------------------------
# vocab из train
VOCAB = ['<blank>', ' ', 'а', 'в', 'д', 'е', 'и', 'к', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'ц', 'ч', 'ш', 'ы', 'ь', 'я']
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for c, i in stoi.items()}


def text_to_ids(text: str) -> List[int]:
    text = normalize_ru_text(text)
    return [stoi[c] for c in text if c in stoi]


def ids_to_text(ids: List[int]) -> str:
    s = "".join(itos[i] for i in ids if i in itos and itos[i] != "<blank>")
    return normalize_ru_text(s)

# --------------------------------------------------------
# audio loading
def load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    path = Path(path)
    assert path.exists(), f"Missing audio file: {path}"

    last_err = None
    for backend in ["ffmpeg", "soundfile", None]:
        try:
            if backend is None:
                wav, sr = torchaudio.load(str(path))
            else:
                wav, sr = torchaudio.load(str(path), backend=backend)
            # wav: [channels, time]
            if wav.dim() != 2:
                raise RuntimeError(f"Unexpected wav shape: {wav.shape}")
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav, sr
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load audio: {path}. Last error: {last_err}")

# --------------------------------------------------------
# feature extraction
@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400         # 25 ms @ 16k
    win_length: int = 400
    hop_length: int = 160    # 10 ms @ 16k
    f_min: float = 20.0
    f_max: Optional[float] = 8000.0
    apply_cmvn: bool = True

@dataclass
class SpecAugmentConfig:
    enabled: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 35
    num_freq_masks: int = 2
    num_time_masks: int = 2

class LogMelFeatureExtractor(nn.Module):
    def __init__(self, feat_cfg: FeatureConfig, spec_cfg: SpecAugmentConfig):
        super().__init__()
        self.feat_cfg = feat_cfg
        self.spec_cfg = spec_cfg

        self.mel = T.MelSpectrogram(
            sample_rate=feat_cfg.sample_rate,
            n_fft=feat_cfg.n_fft,
            win_length=feat_cfg.win_length,
            hop_length=feat_cfg.hop_length,
            f_min=feat_cfg.f_min,
            f_max=feat_cfg.f_max,
            n_mels=feat_cfg.n_mels,
            power=2.0,
            center=True,
        )

        self.freq_mask = T.FrequencyMasking(freq_mask_param=spec_cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=spec_cfg.time_mask_param)

    def forward(self, wav_16k: torch.Tensor, train: bool) -> torch.Tensor:
        """
        wav_16k: [1, T]
        returns feats: [T_frames, n_mels]
        """
        with torch.no_grad():
            mel = self.mel(wav_16k)                 # [1, n_mels, frames]
            feat = torch.log(mel + 1e-5)            # [1, n_mels, frames]

            if train and self.spec_cfg.enabled:
                x = feat
                for _ in range(self.spec_cfg.num_freq_masks):
                    x = self.freq_mask(x)
                for _ in range(self.spec_cfg.num_time_masks):
                    x = self.time_mask(x)
                feat = x

            # [1, n_mels, frames] -> [frames, n_mels]
            feat = feat.squeeze(0).transpose(0, 1).contiguous()

            if self.feat_cfg.apply_cmvn:
                mean = feat.mean(dim=0, keepdim=True)
                std = feat.std(dim=0, keepdim=True).clamp_min(1e-5)
                feat = (feat - mean) / std

            return feat.float()


# --------------------------------------------------------
# model
def conv_out_len(L: torch.Tensor, kernel: int, stride: int, pad: int, dilation: int = 1) -> torch.Tensor:
    return torch.floor((L + 2*pad - dilation*(kernel - 1) - 1) / stride + 1).to(torch.long)

class SmallConformerCTC(nn.Module):
    """
    Lightweight Conformer-CTC:
      log-mel (80) -> 1D conv subsampling (x4) -> Conformer -> Linear -> CTC
    """
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 80,
        d_model: int = 160,
        num_layers: int = 10,
        num_heads: int = 4,
        ffn_dim: int = 320,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        k = 5
        p = k // 2
        self.subsample = nn.Sequential(
            nn.Conv1d(feat_dim, d_model, kernel_size=k, stride=2, padding=p),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k, stride=2, padding=p),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sub_k = k
        self.sub_p = p

        self.encoder = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )

        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feats:     [B, T, 80]
        feat_lens: [B]
        returns:
          logits:    [B, T', vocab]
          out_lens:  [B]
        """
        x = feats.transpose(1, 2)  # [B, 80, T]
        x = self.subsample(x)      # [B, d_model, T']
        out_lens = feat_lens
        out_lens = conv_out_len(out_lens, kernel=self.sub_k, stride=2, pad=self.sub_p)
        out_lens = conv_out_len(out_lens, kernel=self.sub_k, stride=2, pad=self.sub_p)

        x = x.transpose(1, 2).contiguous()  # [B, T', d_model]
        x, out_lens = self.encoder(x, out_lens)  # conforms to torchaudio Conformer forward

        logits = self.ctc_head(x)  # [B, T', vocab]
        return logits, out_lens
    

# --------------------------------------------------------
#  decoding
@torch.no_grad()
def ctc_greedy_decode_batch(logits: torch.Tensor, out_lens: torch.Tensor, blank_id: int = 0) -> List[str]:
    """
    logits:   [B, T, V]
    out_lens: [B]
    returns list of decoded strings (RU text)
    """
    preds = logits.argmax(dim=-1).cpu().tolist()  # [B, T]
    out_lens = out_lens.cpu().tolist()

    texts = []
    for seq, L in zip(preds, out_lens):
        seq = seq[:L]
        collapsed = []
        prev = None
        for t in seq:
            if t != prev:
                collapsed.append(t)
            prev = t
        # remove blanks
        collapsed = [t for t in collapsed if t != blank_id]
        texts.append(ids_to_text(collapsed))
    return texts


def ru_text_to_digits_str(pred_ru: str) -> str:
    """
    Convert model's RU text prediction -> digits string
    If parsing fails, return "" (gives CER=1 vs non-empty ref)
    """
    pred_ru = normalize_ru_text(pred_ru)
    if not pred_ru:
        return ""
    try:
        n = ru_words_to_int(pred_ru)
        n = int(max(0, min(999_999, n)))
        return str(n)
    except Exception:
        return ""


# --------------------------------------------------------
#  high level inference
def load_model_from_checkpoint(ckpt_url_or_path: str, device: str = "cuda") -> Tuple[nn.Module, LogMelFeatureExtractor]:
    """
    Loads model and feature extractor from a checkpoint file.
    Supports local path or URL (via torch.hub.load_state_dict_from_url).
    """
    if ckpt_url_or_path.startswith("http"):
        ckpt_path = torch.hub.load_state_dict_from_url(ckpt_url_or_path, map_location=device, check_hash=False)
    else:
        ckpt_path = ckpt_url_or_path
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab = checkpoint.get("vocab", VOCAB)
    
    feat_cfg = FeatureConfig(**checkpoint["config"]["feat_cfg"])
    spec_cfg = SpecAugmentConfig(enabled=False)
    feature_extractor = LogMelFeatureExtractor(feat_cfg, spec_cfg).to(device)
    feature_extractor.eval()
    
    model = SmallConformerCTC(
        vocab_size=len(vocab),
        feat_dim=feat_cfg.n_mels,
        d_model=160,
        num_layers=10,
        num_heads=4,
        ffn_dim=320,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, feature_extractor


def predict_audio_file(model, feature_extractor, audio_path: Path, device: str = "cuda") -> str:
    """
    Loads audio, extracts features, runs model, decodes to digits string.
    """
    wav, sr = load_audio_mono(audio_path)
    wav = wav.float().to(device)
    # resample to 16k if needed
    if sr != 16000:
        resampler = T.Resample(sr, 16000).to(device)
        wav = resampler(wav)
    feats = feature_extractor(wav, train=False)  # [T, 80]
    feats = feats.unsqueeze(0)  # [1, T, 80]
    feat_len = torch.tensor([feats.size(1)], device=device)
    
    with torch.no_grad():
        logits, out_lens = model(feats, feat_len)
    pred_ru = ctc_greedy_decode_batch(logits, out_lens, blank_id=0)[0]
    digits = ru_text_to_digits_str(pred_ru)
    return digits