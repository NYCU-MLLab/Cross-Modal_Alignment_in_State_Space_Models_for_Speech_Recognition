import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, ConcatDataset

class FastSlidingCMVN(nn.Module):
    """GPU 版滑動 CMVN，會自動處理短於視窗的序列。"""
    def __init__(self, cmn_window: int = 301):
        super().__init__()
        self.K = cmn_window | 1          # force odd
        kernel = torch.ones(1, 1, self.K) / self.K
        self.register_buffer("kernel", kernel)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [B,T,F]
        B, T, C = feats.shape
        x = feats.transpose(1, 2).reshape(B * C, 1, T)       # [B*C,1,T]

        # 若序列短於視窗 => 直接做句級 CMVN
        if T < self.K:
            mu  = x.mean(dim=2, keepdim=True)
            std = x.var(dim=2, unbiased=False, keepdim=True).clamp(1e-8).sqrt()
            norm = (x - mu) / std
        else:
            pad = self.K // 2
            # 對極短序列 (T==1) 改用 replicate，避免 reflect 失敗
            pad_mode = "reflect" if T > 1 else "replicate"
            x_pad = F.pad(x, (pad, pad), mode=pad_mode)
            mu  = torch.conv1d(x_pad, self.kernel)
            var = torch.conv1d(x_pad.pow(2), self.kernel) - mu.pow(2)
            std = var.clamp(1e-8).sqrt()
            norm = (x - mu) / std

        return norm.reshape(B, C, T).transpose(1, 2)         # [B,T,F]

# ---------- Wave → logMel → CMVN → Aug --------------------- #
class Wave2Fbank(nn.Module):
    def __init__(self, train=True):
        super().__init__()
        self.speed = T.SpeedPerturbation(16000, [0.9, 1.0, 1.1]) if train else None
        self.mel   = T.MelSpectrogram(sample_rate=16000, n_mels=80)
        self.cmvn  = FastSlidingCMVN(301)
        self.fmask = T.FrequencyMasking(27) if train else None
        self.tmask = T.TimeMasking(15, p=0.05) if train else None

    def forward(self, wav):                       # wav:  [1, time] (CPU)
        if self.speed is not None:
            wav, _ = self.speed(wav)

        power = self.mel(wav).clamp_min(1e-5)   # [1,80,T]
        fb = 10.0 * power.log10()
        fb = fb.transpose(1, 2)                      # → [1,T,80]
        fb = self.cmvn(fb)
        fb = fb.transpose(1, 2)                      # → [1,80,T]
        if self.fmask is not None:
            fb = self.fmask(fb)
            fb = self.tmask(fb)
        return fb

def data_processing(data, data_type="train", encoder_type=None, tokenizer=None, train_audio_transforms=None, valid_audio_transforms=None):
    spectrograms = []
    labels = []
    ar_tgts = []
    ar_labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        
        spectrograms.append(spec)
        label = torch.tensor(tokenizer.text_to_int(utterance.lower()), dtype=torch.long)

        # add <sos> (1) in the label
        ar_tgt = torch.cat((torch.tensor([tokenizer.sp.bos_id()]), label))
        # add <eos> (2) in the label
        ar_label = torch.cat((label, torch.tensor([tokenizer.sp.eos_id()])))
        
        labels.append(label)
        ar_tgts.append(ar_tgt)
        ar_labels.append(ar_label)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    if encoder_type == "mamba":
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0, padding_side='left').unsqueeze(1).transpose(2, 3) # mamba
    elif encoder_type == "conformer":
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0).unsqueeze(1).transpose(2, 3) # conformer

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.sp.piece_to_id("<pad>"))
    ar_tgts = nn.utils.rnn.pad_sequence(ar_tgts, batch_first=True, padding_value=tokenizer.sp.piece_to_id("<pad>"))
    ar_labels = nn.utils.rnn.pad_sequence(ar_labels, batch_first=True, padding_value=tokenizer.sp.piece_to_id("<pad>"))

    return spectrograms, labels, ar_tgts, ar_labels, input_lengths, label_lengths

def get_dataloaders(tokenizer, config):
    dataset_dir = "/home/fanche/thesis/work/dataset"
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    # for Librispeech
    splits = ["train-clean-100", "train-clean-360", "train-other-500"]

    test_clean_url="test-clean"
    test_other_url="test-other"

    dev_clean_url="dev-clean"
    dev_other_url="dev-other"

    train_dataset100 = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=splits[0], download=False)
    train_dataset360 = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=splits[1], download=False)
    train_dataset500 = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=splits[2], download=True)

    test_clean_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=test_clean_url, download=False)
    test_other_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=test_other_url, download=False)

    dev_clean_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=dev_clean_url, download=False)
    dev_other_dataset = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=dev_other_url, download=False)

    # Combine the dataset splits into a single dataset
    # train_dataset460 = data.ConcatDataset([train_dataset100, train_dataset360])
    train_dataset960 = data.ConcatDataset([train_dataset100, train_dataset360, train_dataset500])
    # all_dataset = data.ConcatDataset([train_dataset460, train_dataset500, test_clean_dataset, test_other_dataset, dev_clean_dataset, dev_other_dataset])

    # for TEDLIUM
    release = 'release3'
    splits = ["train", "dev", "test"]

    ted_train_dataset = torchaudio.datasets.TEDLIUM(dataset_dir, release=release, subset=splits[0], download=False)
    ted_dev_dataset = torchaudio.datasets.TEDLIUM(dataset_dir, release=release, subset=splits[1], download=False)
    ted_test_dataset = torchaudio.datasets.TEDLIUM(dataset_dir, release=release, subset=splits[2], download=False)

    ted_plus_librispeech_dataset = data.ConcatDataset([ted_train_dataset, train_dataset100, train_dataset360, train_dataset500])
    
    # --- 實例化 Audio Transforms ---
    train_audio_transforms = Wave2Fbank(train=True)
    valid_audio_transforms = Wave2Fbank(train=False)

    # --- 建立 DataLoaders ---
    # (將所有 DataLoader 的定義移到這裡)
    train_loader = DataLoader(...)
    dev_loader = DataLoader(...)
    # ...
    
    return train_loader, dev_loader