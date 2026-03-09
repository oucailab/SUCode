import torch
import torch.backends
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import pdb
import math
import functools

from basicsr.utils.registry import ARCH_REGISTRY


from .my_convblock_arch import AttnResBlock, SPAMBlock, LayerNorm2d
from .my_transformer_arch import TransformerLayers
from .femasr_arch import SwinLayers, ResBlock

from .deform_conv_utils import DeformableConv2d
from .vgg_arch import VGGFeatureExtractor
from .RRDB_arch import RRDB, make_layer
from basicsr.losses import build_loss

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, AdaCode_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage # if LQ_stage is True, it means the indices of input has a groundtruth to learn from.
        self.AdaCode_stage = AdaCode_stage
        self.beta = beta 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)
            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean() 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    
    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class WeightPredictor(nn.Module):
    def __init__(self,
                 in_channel,
                 cls,
                 weight_softmax=False,
                 **swin_opts,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(SwinLayers(**swin_opts))
                     
        # weight
        self.blocks.append(nn.Conv2d(in_channel, cls, kernel_size=1))
        if weight_softmax:
            self.blocks.append(nn.Softmax(dim=1))


    def forward(self, input):
        outputs = []
        x = input
        for idx, m in enumerate(self.blocks):
            x = m(x)

        return x

class WeightPredictor_RRDB(nn.Module):
    def __init__(self,
                 in_channel,
                 cls,
                 weight_softmax=False,
                 num_RRDB=6,
                 **swin_opts,
                 ):
        super().__init__()
        nc = 64

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv2d(in_channel, nc, 3, 1, 1, bias=True))

        RRDB_block_f = functools.partial(RRDB, nf=nc, gc=32)
        self.blocks.append(make_layer(RRDB_block_f, num_RRDB))

        # weight
        self.blocks.append(nn.Conv2d(nc, cls, 3, 1, 1, bias=True))
        if weight_softmax:
            self.blocks.append(nn.Softmax(dim=1))


    def forward(self, input):
        outputs = []
        x = input
        for idx, m in enumerate(self.blocks):
            x = m(x)

        return x

class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out



class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1)
        self.dcn = DeformableConv2d(in_channel, in_channel, 3, stride=1, padding=1)

    def forward(self, x_vq, x_residual):
        x_residual = self.offset(torch.cat([x_vq, x_residual], dim=1))
        feat_after_warp = self.dcn(x_residual)

        return feat_after_warp
    
class FuseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.enc_conv = nn.Sequential(
                            nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1),
                            ResBlock(in_channel, in_channel))

        self.scale = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0))

    def forward(self, x, x_coder, weight=1.0):
        x = self.enc_conv(torch.cat([x, x_coder], dim=1))

        x_scale = self.scale(x)
        x_shift = self.shift(x)

        residual = weight * (x_scale * x_coder + x_shift)
        x = x + residual

        return x
    

class FreqFuseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1)

        self.norm1 = LayerNorm2d(in_channel)
        self.freq_conv_expand = 2
        self.freq_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel * self.freq_conv_expand, kernel_size=1, padding=0),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(in_channel * self.freq_conv_expand, in_channel, kernel_size=1, padding=0))
        self.gamma = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, out_channel, 1, 1)), requires_grad=True)

        self.scale = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))


    def forward(self, x, x_coder, weight=1.0):
        x_in = self.in_conv(torch.cat([x, x_coder], dim=1))
        x_in = self.norm1(x_in)

        # Freq attention 
        # Reference : https://github.com/cidautai/DarkIR/blob/main/archs/arch_model.py#L36
        _, _, H, W = x_in.shape
        x_freq = torch.fft.rfft2(x_in, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.freq_conv(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_freq = torch.complex(real, imag)
        x_freq = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')      

        x_out = x_in * x_freq
        x_out = x_in + self.gamma * x_out

        x_scale = self.scale(x_out)
        x_shift = self.shift(x_out)

        residual = weight * (x_scale * x + x_shift)
        x = x + residual

        return x



class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channel, max_depth, input_res=256, channel_query_dict=None, 
                 norm_type='gn', act_type='leakyrelu', LQ_stage=False, **swin_opts,):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res

        ### Encoder blocks.
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        ### LQ stage blocks.
        if LQ_stage:
            ### Middle blocks (Transformer blocks).
            self.blocks.append(SwinLayers(**swin_opts))

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)
        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)
        return outputs
    

class MultiScaleDecoder(nn.Module):
    def __init__(self, in_channel, max_depth, input_res=256, channel_query_dict=None, 
                 norm_type='gn', act_type='leakyrelu',
                 with_coder=True, with_warp=True):
        super().__init__()
        self.with_coder = with_coder        # use code features from decoder.
        self.with_warp = with_warp          # use warping to fusion the features with code features from decoder.

        self.upsampler = nn.ModuleList()
        if with_warp:
            self.warp = nn.ModuleList()
        # res = input_res // (2 ** max_depth)

    
        for i in range(max_depth):
            res = input_res // 2**max_depth * 2**i
            in_channel, out_channel = channel_query_dict[res] * 2, channel_query_dict[res * 2]
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                ResBlock(out_channel, out_channel, norm_type, act_type),
            ))
            if with_warp:
                self.warp.append(FreqFuseBlock(out_channel, out_channel))

            # res = res * 2

    def forward(self, input, coder_input=None):
        x = input[0]
            
        for idx, m in enumerate(self.upsampler):
                x = torch.cat([x, input[idx+1]], dim=1) ### Concatenate features from encoder.
                x = m(x)

                if coder_input is not None: ### Merge input from LQ decoder.
                    if self.with_warp:
                        x = self.warp[idx](x, coder_input[idx])
                    else:
                        x = x + coder_input[idx] * (x.mean() / coder_input[idx].mean())
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)

    
@ARCH_REGISTRY.register()
class SUCode(nn.Module):
    def __init__(self,
                 *,
                 batch_size,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 AdaCode_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=1,
                 use_semantic_loss=False,
                 use_residual=True,
                 weight_softmax=False,
                 with_warp=True,
                 Coder_stage=False,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.AdaCode_stage = AdaCode_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual
        self.weight_softmax = weight_softmax
        self.batch_size = batch_size

        ### decoder fusion parameters
        self.with_warp = with_warp
        self.with_coder = Coder_stage

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder 
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, LQ_stage
                            )
        if self.LQ_stage:
            self.multiscale_decoder = MultiScaleDecoder(
                                    in_channel,
                                    self.max_depth,
                                    self.gt_res // self.scale_factor,
                                    channel_query_dict,
                                    norm_type, act_type,
                                    with_coder=self.with_coder,
                                    with_warp=self.with_warp
                                )


        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.out_coder_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build weight predictor
        self.weight_predictor = WeightPredictor(
                                channel_query_dict[self.codebook_scale[0]],
                                self.codebook_scale.shape[0],
                                self.weight_softmax
                                )

        # build multi-scale vector quantizers 
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for i in range(codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[i],
                codebook_emb_dim[i],
                LQ_stage=self.LQ_stage,
                AdaCode_stage=self.AdaCode_stage
                )
            self.quantize_group.append(quantize)

            quant_in_ch = channel_query_dict[self.codebook_scale[i]]

            if i == 0:
                comb_quant_in_ch1 = codebook_emb_dim[i]
                comb_quant_in_ch2 = 0
            else:
                comb_quant_in_ch1 = codebook_emb_dim[i]
                comb_quant_in_ch2 = codebook_emb_dim[i]


            self.before_quant_group.append(nn.Conv2d(quant_in_ch, codebook_emb_dim[i], 1))
            self.after_quant_group.append(nn.Conv2d(codebook_emb_dim[i], quant_in_ch, 3, 1, 1))


        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(codebook_emb_dim[-1], 512, 1, 1, 0),
                nn.ReLU(),
                )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer]) 

        # soft label cross entropy loss for weight
        self.weight_cri = build_loss({'type': 'SoftCrossEntropy'})

        # Contrastive loss
        # self.contrast_cri = build_loss({'type': 'ContrastiveLoss', 'batch_size': self.batch_size})

    def encode_and_decode(self, input, gt_aux=None, current_iter=None):
        enc_feats = self.multiscale_encoder(input.detach())
        enc_feats = enc_feats[::-1] # reverse the order of features

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        
        coder_list = []             # features from decoder (for DecoderBlock and MultiScaleDecoder).
        weight = None               # weight for each codebook (for WeightPredictor).

        dec_feat = None
        x = enc_feats[0] 
        
        for i in range(self.max_depth):
            quant_feat_group = []         # after quantized features for each codebook.
            for idx in range(self.codebook_scale.shape[0]):
                quant_feat_group.append(None)

            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # this enc_feat[i] needs to perform quantize
                
                if dec_feat is not None:
                    before_quant_feat = torch.cat((x, dec_feat), dim=1)
                else:
                    before_quant_feat = x

                # quantize features with multiple codebooks
                for codebook_idx in range(self.codebook_scale.shape[0]):
                    feat_to_quant = self.before_quant_group[codebook_idx](before_quant_feat)
                    if gt_aux is not None:
                        z_quant, codebook_loss, indices = self.quantize_group[codebook_idx](feat_to_quant, gt_aux['indices'][codebook_idx])
                    else:
                        z_quant, codebook_loss, indices = self.quantize_group[codebook_idx](feat_to_quant)
                    if not self.use_quantize:
                        z_quant = feat_to_quant

                    after_quant_feat = self.after_quant_group[codebook_idx](z_quant)

                    quant_feat_group[codebook_idx] = after_quant_feat
                    codebook_loss_list.append(codebook_loss)
                    indices_list.append(indices)

                # merge feature tensors
                weight = self.weight_predictor(before_quant_feat).unsqueeze(2) # B x N x 1 x H x W
                x = torch.sum(torch.mul(torch.stack(quant_feat_group).transpose(0, 1), weight), dim=1)
                feat_before_decoder = x

            # decoder block for LQ image.
            x = self.decoder_group[i](x)
            coder_list.append(x)
            dec_feat = x

        out_img = self.out_conv(x)
        ### Multi-scale decoder for out image.
        if self.LQ_stage and self.with_coder:
            coder_feature = self.multiscale_decoder(enc_feats, coder_list)
            out_img = self.out_coder_conv(coder_feature)


        # loss
        loss_dict = {}
        loss_dict['codebook_loss'] = sum(codebook_loss_list)
        loss_dict['semantic_loss'] = sum(semantic_loss_list) if len(semantic_loss_list) else loss_dict['codebook_loss'] * 0
        if gt_aux is not None:
            if self.LQ_stage:
                loss_dict['contrast_loss'] = self.contrast_cri(before_quant_feat, gt_aux['feat_before_quant'])
            loss_dict['before_quant_loss'] = self.feature_loss(before_quant_feat, gt_aux['feat_before_quant']) # <before_quant_feat, gt_aux['feat_before_quant']>
            loss_dict['after_quant_loss'] = self.feature_loss(feat_before_decoder, gt_aux['feat_before_decoder']) # <feat_before_decoder, gt_aux['feat_before_decoder']>
            # soft label cross entropy for weight, can be used only when weight_softmax is True
            if self.weight_softmax:
                loss_dict['weight_loss'] = self.weight_cri(weight.squeeze(2), gt_aux['weight'].squeeze(2))
        # auxilary
        aux = {}
        aux['weight'] = weight.detach()
        aux['indices'] = indices_list
        aux['feat_before_decoder'] = feat_before_decoder.detach()
        aux['feat_before_quant'] = before_quant_feat.detach()

        return out_img, loss_dict, aux

    def feature_loss(self, z, z_gt):
        """
        Args:
            z: lq features BxCxHxW
            z_gt: gt features BxCxHxW
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_gt = z_gt.permute(0, 2, 3, 1).contiguous()
        feature_loss = 0.25 * ((z_gt.detach() - z)**2).mean()
        feature_loss += self.gram_loss(z, z_gt.detach())
        return feature_loss

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()
    
    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16, vis_weight=False):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        weight_map = torch.zeros(batch, len(self.codebook_scale), 1, output_height, output_width)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                if vis_weight:
                    output_tile, weight_tile = self.test(input_tile, vis_weight=True)
                else:
                    output_tile = self.test(input_tile)
                
                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
                if vis_weight:
                    weight_map[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = weight_tile[:, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        
        if vis_weight:
            return output, weight_map
        return output

    @torch.no_grad()
    def test(self, input, vis_weight=False):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 8 // self.scale_factor * 8 
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        dec, _, aux = self.encode_and_decode(input)

        output = dec
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]

        self.use_semantic_loss = org_use_semantic_loss 

        if vis_weight:
            weight = aux['weight']
            weight = F.pixel_shuffle(weight.repeat(1,1,64,1,1), 8)
            weight = weight[..., :h_old * self.scale_factor, :w_old * self.scale_factor]
            return output, weight
        return output

    def forward(self, input, gt_aux=None):

        if gt_aux is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, losses, aux = self.encode_and_decode(input, gt_aux)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, losses, aux = self.encode_and_decode(input)

        return dec, losses, aux
