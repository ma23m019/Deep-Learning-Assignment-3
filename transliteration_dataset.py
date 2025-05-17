%%writefile transliteration_dataset.py

import torch
from torch.utils.data import Dataset
from collections import Counter
from torchtext.vocab import Vocab
import unicodedata

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def normalize(text):
    return unicodedata.normalize('NFKC', text.strip())

def read_data(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
    latins, targets = [], []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            latin, target = parts[:2]
            latins.append(normalize(latin))
            targets.append(normalize(target))
    return latins, targets

def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text)
    return Vocab(counter, specials=SPECIAL_TOKENS)

def encode(word, vocab):
    return [vocab[SOS_TOKEN]] + [vocab[char] for char in word] + [vocab[EOS_TOKEN]]

def decode(indices, vocab):
    inv_vocab = {v: k for k, v in vocab.get_stoi().items()}
    return ''.join(inv_vocab[i] for i in indices if i >= len(SPECIAL_TOKENS))

class TransliterationDataset(Dataset):
    def __init__(self, latins, targets, src_vocab, tgt_vocab):
        self.data = list(zip(latins, targets))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        latin, target = self.data[idx]
        src_ids = torch.tensor(encode(latin, self.src_vocab))
        tgt_ids = torch.tensor(encode(target, self.tgt_vocab))
        return src_ids, tgt_ids

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_pad, tgt_pad
