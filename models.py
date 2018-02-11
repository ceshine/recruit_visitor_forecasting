"""End-to-end Pytorch Models"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from components import (
    LockedDropout, TransformerEncoder, TransformerDecoder
)
from transformer.Modules import LayerNormalization

TRAIN_PERIODS = 140


class BaseModel(nn.Module):
    def _create_layers(self, mlp=False):
        self.store_area_em = nn.Embedding(103, 10, max_norm=10, norm_type=2)
        self.store_municipal_em = nn.Embedding(55, 5, max_norm=5, norm_type=2)
        self.store_prefecture_em = nn.Embedding(9, 2, max_norm=2, norm_type=2)
        self.store_genre_em = nn.Embedding(14, 5, max_norm=5, norm_type=2)
        # self.weekday_em = nn.Embedding(7, 5, max_norm=5, norm_type=2)
        self.day_em = nn.Embedding(31, 5, max_norm=5, norm_type=2)
        # self.month_em = nn.Embedding(12, 5, max_norm=5, norm_type=2)
        if not mlp:
            self.step_one_network = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                LayerNormalization(self.hidden_size),
                nn.Dropout(self.odrop),
                nn.Linear(self.hidden_size, 1)
            )

    def __init__(self, *, bidirectional, edrop, odrop, propagate, min_length, y_scale_by, steps=38):
        super(BaseModel, self).__init__()
        # numeric + derived + onehot + embeddings
        self.propagate = propagate
        self.input_dim = 5 + 5 + 4 + 7 + (5 + 5 + 10 + 2 + 5)
        self.decode_dim = (
            self.input_dim - 2 - int(not self.propagate) + steps)
        # -2: is_zero, reservation_from
        self.odrop = odrop
        self.bidirectional = bidirectional
        self.edrop = edrop
        self.min_length = min_length
        self.steps = steps
        self.y_scale_by = y_scale_by
        self.edropout = LockedDropout(batch_first=True)

    def init_weights(self, mlp=False):
        nn.init.kaiming_normal(self.store_genre_em.weight)
        nn.init.kaiming_normal(self.store_area_em.weight)
        nn.init.kaiming_normal(self.store_prefecture_em.weight)
        nn.init.kaiming_normal(self.store_municipal_em.weight)
        # nn.init.orthogonal(self.month_em.weight)
        nn.init.kaiming_normal(self.day_em.weight)
        # nn.init.kaiming_normal(self.year_em.weight)
        if not mlp:
            for m in self.step_one_network:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal(m.weight)
                    nn.init.constant(m.bias, 0)

    def preprocess_x(self, x, x_d, x_i):
        embeddings = torch.cat([
            # self.weekday_em((x_i[:, :, 1]).long()),
            self.day_em((x_i[:, :, 2]).long()),
            # self.month_em((x_i[:, :, 3]).long()),
            self.store_genre_em(x_i[:, :, 4].long()),
            self.store_prefecture_em(x_i[:, :, 5].long()),
            self.store_area_em(x_i[:, :, 6].long()),
            self.store_municipal_em(x_i[:, :, 7].long()),
        ], dim=2)
        if self.edrop:
            embeddings = self.edropout(embeddings, self.edrop)
        if self.training and self.min_length != TRAIN_PERIODS:
            start = random.randint(0, TRAIN_PERIODS - self.min_length + 1)
        else:
            start = 0
        weekday_idx = torch.cuda.FloatTensor(
            x.size()[0], TRAIN_PERIODS + self.steps, 7).zero_()
        weekday_idx.scatter_(2, x_i[:, :, 1:2].data.cuda().long(), 1)
        x_encode = torch.cat([
            x[:, start:TRAIN_PERIODS, :].float(),
            embeddings[:, start:TRAIN_PERIODS, :],
            # Derived features
            x_d[:, :].unsqueeze(1).expand(
                x_d.size()[0], TRAIN_PERIODS - start, x_d.size()[1]).float(),
            # Holiday
            x_i[:, start:TRAIN_PERIODS, :1].float(),
            # Is ZERO + IS NULL x2
            x_i[:, start:TRAIN_PERIODS, -3:].float(),
            # Weekday
            Variable(weekday_idx[:, start:TRAIN_PERIODS, :].float())
        ], dim=2)
        step_idx = torch.cuda.FloatTensor(
            x.size()[0], self.steps, self.steps).zero_()
        step_idx.scatter_(
            2, torch.arange(0, self.steps).cuda().long(
            ).unsqueeze(0).expand(x.size()[0], self.steps).unsqueeze(2), 1)
        x_decode = torch.cat([
            embeddings[:, TRAIN_PERIODS:, :],
            x[:, TRAIN_PERIODS:, 2:].float(),
            # Derived features
            x_d[:, :].unsqueeze(1).expand(
                x_d.size()[0], self.steps, x_d.size()[1]),
            # IS NULL x2
            x_i[:, TRAIN_PERIODS:, -2:].float(),
            # Holiday
            x_i[:, TRAIN_PERIODS:, :1].float(),
            Variable(step_idx).float(),
            Variable(weekday_idx[:, TRAIN_PERIODS:, :].float())
        ], dim=2)
        if self.propagate:
            x_decode = torch.cat([
                x[:, TRAIN_PERIODS:, :1], x_decode], dim=2)
        return x_encode, x_decode


class TransformerModel(BaseModel):
    def __init__(
            self, n_max_seq, *,  y_scale_by, steps, min_length, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            edrop=0.25, odrop=0.25, hdrop=0.1, propagate=False):
        super(TransformerModel, self).__init__(
            bidirectional=False, edrop=edrop, odrop=odrop, propagate=propagate,
            min_length=min_length, y_scale_by=y_scale_by, steps=steps
        )
        self.hidden_size = d_model
        self._create_layers(mlp=True)
        self.encoder = TransformerEncoder(
            n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=hdrop)
        self.decoder = TransformerDecoder(
            n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=hdrop)
        self.encoder_mapping = nn.Linear(self.input_dim, d_word_vec)
        self.decoder_mapping = nn.Linear(self.decode_dim, d_word_vec)
        self.step_one_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            LayerNormalization(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.output_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            LayerNormalization(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.init_linear_weights()

        assert d_model == d_word_vec
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

    def init_linear_weights(self):
        """Class specific initialization"""
        super(TransformerModel, self).init_weights(mlp=True)
        nn.init.orthogonal(self.encoder_mapping.weight)
        nn.init.constant(self.encoder_mapping.bias, 0)
        nn.init.orthogonal(self.decoder_mapping.weight)
        nn.init.constant(self.decoder_mapping.bias, 0)
        for submodel in (self.encoder, self.decoder):
            for m in submodel.parameters():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.orthogonal(m.weight)
                    nn.init.constant(m.bias, 0)
        for submodel in (self.output_network, self.step_one_network):
            for m in submodel:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal(m.weight)
                    nn.init.constant(m.bias, 0)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(
            map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(
            map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, x, x_d, x_i, *, tf_ratio=0, **ignore):
        x_encode, x_decode = self.preprocess_x(x, x_d, x_i)
        src_pos = Variable(torch.arange(0, x_encode.size()[1]).cuda().long())
        tgt_pos = Variable(torch.arange(0, x_decode.size()[1]).cuda().long())
        x_encode = self.encoder_mapping(x_encode)
        enc_output, *_ = self.encoder(x_encode, src_pos)
        step_one = self.step_one_network(
            F.dropout(enc_output[:, -1, :], self.odrop))
        if self.propagate:
            output = Variable(
                torch.cuda.FloatTensor(x.size()[0], self.steps + 1).zero_())
            previous = step_one
            output[:, 0] = previous[:, 0]
            for j in range(self.steps):
                if random.random() >= tf_ratio:
                    x_decode[:, j, 0] = previous[:, 0].detach()
                dec_output, *_ = self.decoder(
                    self.decoder_mapping(
                        x_decode[:, :(j + 1), :]), tgt_pos[:(j + 1)],
                    x_encode, enc_output)
                previous = self.output_network(
                    F.dropout(dec_output[:, -1, :], self.odrop))
                output[:, j + 1] = previous[:, 0]
            return output
        x_decode = self.decoder_mapping(x_decode)
        dec_output, *_ = self.decoder(x_decode, tgt_pos, x_encode, enc_output)
        reg_output = self.output_network(
            F.dropout(dec_output, self.odrop)).squeeze(2)
        return torch.cat([step_one, reg_output], dim=1)
