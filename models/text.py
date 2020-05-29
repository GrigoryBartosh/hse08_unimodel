import copy

import torch
import torch.nn as nn

from transformers import BertModel
from sru import SRU

from models.common import mlp_by_name

__all__ = ['Bert', 'TextEncoder', 'TextDecoder', 'TextDiscriminator']


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text, mask=None):
        x = self.bert(text, attention_mask=mask)[0]
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional
        )

        shape = ((2 if bidirectional else 1) * num_layers, hidden_size)
        self.hidden = nn.Parameter(torch.randn(shape))
        self.state = nn.Parameter(torch.randn(shape))

    def forward(self, x, mask):
        bs = x.shape[0]

        h_n = self.hidden[:, None, :].expand(-1, bs, -1)
        c_n = self.state[:, None, :].expand(-1, bs, -1)

        length = mask.sum(axis=1)
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True,
                                              enforce_sorted=False)
        _, (_, c_n) = self.lstm(x, (h_n, c_n))

        c_n = c_n.sum(axis=0)

        return c_n


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(LSTMDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional
        )

        shape = ((2 if bidirectional else 1) * num_layers, self.hidden_size)
        self.hidden = nn.Parameter(torch.randn(shape))

        self.input = nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, c_n, max_len):
        bs = c_n.shape[0]

        h_n = self.hidden
        h_n = self.hidden[:, None, :].expand(-1, bs, -1).contiguous()

        c_n = c_n[None, :, :].expand_as(h_n).contiguous()

        x = self.input
        x = x[None, None, :].expand(bs, max_len, -1)

        x, _ = self.lstm(x, (h_n, c_n))

        x = x.reshape(bs, max_len, -1, self.hidden_size)
        x = x.sum(axis=2)

        return x


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(GRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional
        )

        shape = ((2 if bidirectional else 1) * num_layers, hidden_size)
        self.hidden = nn.Parameter(torch.randn(shape))

    def forward(self, x, mask):
        bs = x.shape[0]

        h_n = self.hidden[:, None, :].expand(-1, bs, -1)

        length = mask.sum(axis=1)
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True,
                                              enforce_sorted=False)
        _, h_n = self.gru(x, h_n)

        h_n = h_n.sum(axis=0)

        return h_n


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(GRUDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=self.bidirectional
        )

        self.input = nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, h_n, max_len):
        bs = h_n.shape[0]

        h_n = h_n[None, :, :].expand(
        	(2 if self.bidirectional else 1) * self.num_layers, -1, -1)
        h_n = h_n.contiguous()

        x = self.input
        x = x[None, None, :].expand(bs, max_len, -1)

        x, _ = self.gru(x, h_n)

        x = x.reshape(bs, max_len, -1, self.hidden_size)
        x = x.sum(axis=2)

        return x


class SRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(SRUEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.sru = SRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout_p,
            rnn_dropout=dropout_p,
            bidirectional=bidirectional
        )

        shape = (self.num_layers, (2 if bidirectional else 1) * self.hidden_size)
        self.state = nn.Parameter(torch.randn(shape))

    def forward(self, x, mask):
        bs = x.shape[0]

        c_n = self.state[:, None, :].expand(-1, bs, -1)

        mask = (1 - mask).permute(1, 0).byte()
        x = x.permute(1, 0, 2)
        _, c_n = self.sru(x, c_n, mask)

        c_n = c_n.reshape(self.num_layers, bs, -1, self.hidden_size)
        c_n = c_n.sum(axis=[0, 2])

        return c_n


class SRUDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p=0, bidirectional=False):
        super(SRUDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.sru = SRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout_p,
            rnn_dropout=dropout_p,
            bidirectional=self.bidirectional
        )

        self.input = nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, c_n, max_len):
        bs = c_n.shape[0]

        c_n = c_n[None, :, None, :].expand(self.num_layers, -1,
                                           (2 if self.bidirectional else 1), -1)
        c_n = c_n.reshape(self.num_layers, bs, -1)

        x = self.input
        x = x[None, None, :].expand(max_len, bs, -1)

        x, _ = self.sru(x, c_n)

        x = x.reshape(max_len, bs, -1, self.hidden_size)
        x = x.sum(axis=2)
        x = x.permute(1, 0, 2)

        return x


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()

        self.embeds = torch.load(args['embeds_path'])
        self.embeds = nn.Embedding.from_pretrained(self.embeds['weight'])

        if args['rnn'] == 'LSTM':
            rnn_type = LSTMEncoder
        elif args['rnn'] == 'GRU':
            rnn_type = GRUEncoder
        elif args['rnn'] == 'SRU':
            rnn_type = SRUEncoder
        else:
            assert False, f"Unsupported encoder RNN name: {args['rnn']}"

        self.rnn = rnn_type(
            input_size=self.embeds.weight.shape[1],
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            dropout_p=args['dropout_p'],
            bidirectional=args['bidirectional']
        )

    def forward(self, ids, mask):
        x = self.embeds(ids)
        x = self.rnn(x, mask)
        return x


class TextDecoder(nn.Module):
    def __init__(self, args):
        super(TextDecoder, self).__init__()

        if args['rnn'] == 'LSTM':
            rnn_type = LSTMDecoder
        elif args['rnn'] == 'GRU':
            rnn_type = GRUDecoder
        elif args['rnn'] == 'SRU':
            rnn_type = SRUDecoder
        else:
            assert False, f"Unsupported decoder RNN name: {args['rnn']}"

        self.rnn = rnn_type(
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            dropout_p=args['dropout_p'],
            bidirectional=args['bidirectional']
        )

        self.linear = nn.Linear(args['hidden_size'], args['vocab_size'])

    def forward(self, x, max_len):
        x = self.rnn(x, max_len)
        x = self.linear(x)
        return x


class TextEmbedEncoder(nn.Module):
    def __init__(self, args):
        super(TextEmbedEncoder, self).__init__()

        self.mlp = mlp_by_name(args['mlp'])(
            in_dim=args['text_embed_dim'],
            layer_dims=args['layers'],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x):
        return self.mlp(x)


class TextEmbedDecoder(nn.Module):
    def __init__(self, args):
        super(TextEmbedDecoder, self).__init__()

        self.mlp = mlp_by_name(args['mlp'])(
            in_dim=args['latent_dim'],
            layer_dims=args['layers'],
            activ=args['activ'],
            norm_layer=args['norm_layer'],
            dropout_p=args['dropout_p']
        )

    def forward(self, x):
        return self.mlp(x)


class TextEmbedDiscriminator(nn.Module):
    def __init__(self, args):
        super(TextEmbedDiscriminator, self).__init__()

        args = copy.deepcopy(args)
        args['layers'] += [1]
        self.model = TextEmbedEncoder(args)

    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
        return x