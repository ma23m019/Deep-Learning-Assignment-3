%%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder with GRU
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)  # outputs: [batch, src_len, hid_dim]
        return outputs, hidden  # hidden: [1, batch, hid_dim]


# Attention mechanism (Bahdanau-style additive attention)
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.permute(1, 0, 2)  # [batch, 1, dec_hid_dim]
        hidden = hidden.repeat(1, src_len, 1)  # [batch, src_len, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        return F.softmax(attention, dim=1)


# Decoder with Attention
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(input)  # [batch, 1, emb_dim]

        a = self.attention(hidden, encoder_outputs)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]

        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid_dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [batch, 1, emb + enc_hid]

        output, hidden = self.rnn(rnn_input, hidden)  # [batch, 1, dec_hid]
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [batch, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: [batch, src_len]
        # trg: [batch, trg_len]

        encoder_outputs, hidden = self.encoder(src)

        # Ignore <end> token
        decoder_input = trg[:, :-1]  # [batch, trg_len - 1]

        batch_size, trg_len = decoder_input.size()
        outputs = []

        input_token = decoder_input[:, 0]
        hidden = hidden

        for t in range(trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)  # attention-aware decoder
            outputs.append(output.unsqueeze(1))  # [batch, 1, output_dim]

            # teacher forcing: feed ground truth token
            if t + 1 < trg_len:
                input_token = decoder_input[:, t + 1]

        outputs = torch.cat(outputs, dim=1)  # [batch, trg_len, output_dim]
        return outputs
