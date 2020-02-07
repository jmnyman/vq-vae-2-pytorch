import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, n_additional_downsample_layers):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # ADDITIONAL DOWNSAMPLING
        for i in range(n_additional_downsample_layers):
            blocks.append(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, n_additional_upsample_layers):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # additional upsampling to match additional downsampling
        for i in range(n_additional_upsample_layers):
            blocks.append(nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        n_additional_downsample_layers=3,
        n_additional_upsample_layers=3,
        num_classes=2,
        input_size=512,
    ):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, 4, n_additional_downsample_layers)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, 2, 0)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, 2, n_additional_upsample_layers=0
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            4,
            n_additional_upsample_layers
        )

        self.dropout = nn.Dropout()

        self.downsample_top_size = int(self.input_size / 2**(3+n_additional_downsample_layers))
        self.unrolled_top_size = int(self.embed_dim * self.downsample_top_size**2)

        # CLASSIFIER [disabled by default with a scale of 0]
        self.num_classes = num_classes
        self.classifier_fc = nn.Linear(self.embed_dim, self.num_classes) # if not unrolling!
        # self.classifier_fc = nn.Linear(self.unrolled_top_size, self.num_classes)
        self.cross_ent = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, input, labels):
        quant_t, quant_b, diff, _, _, enc_t, enc_b, classifier_loss = self.encode(input, labels)
        dec = self.decode(quant_t, quant_b)

        return dec, diff, enc_t, enc_b, classifier_loss

    def encode(self, input, labels):
        batch_size = input.shape[0]

        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        pre_quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) # pre-actual quantization
        # repeat labels to match enc_t's latent field dimensions [pointwise classification]
        repeat_labels = labels.view(batch_size,1,1).repeat(1, self.downsample_top_size, self.downsample_top_size)
        
        # New: run classifier on pre-quantized top level encoding
#         classifier_logits = self.classifier_fc(self.dropout(pre_quant_t.contiguous().view(batch_size,-1)))
        classifier_logits = self.classifier_fc(self.dropout(pre_quant_t)) # if not unrolling!

        # need unsqueeze(0) for dataparallel formatting
#         classifier_loss = self.cross_ent(classifier_logits, labels.long()).unsqueeze(0)
        # reshape again for crossent
        classifier_loss = self.cross_ent(classifier_logits.permute(0,3,1,2), repeat_labels.long()).unsqueeze(0) 
        
        quant_t, diff_t, id_t = self.quantize_t(pre_quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        pre_quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(pre_quant_b) # I renamed to avoid confusion; not sure performance hit
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b, enc_t, enc_b, classifier_loss

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
