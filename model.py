import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Lost_mask import Lost_Mask__Net_Model
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import copy
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from utils import colorize

os.environ['CUDA_VISIBLE_DEVICE'] = "0"
torch.cuda.set_device(0)
Lost_Mask_model = Lost_Mask__Net_Model()
Lost_Mask_model = Lost_Mask_model.cuda()

fire_dir = "/home/data1/hyl/Checkpoints/11-11-2/"
for root, dirs, files in os.walk(fire_dir,topdown = False):
    print(files)
max_lm_value = 0
max_lm_index = 0
for sep_num in range(len(files)):
    str_list = files[sep_num].split(sep='_')
    if max_lm_value < int(str_list[2]):
        max_lm_value = int(str_list[2])
        max_lm_index = sep_num
model_name =  fire_dir + files[max_lm_index]
print(model_name)
model_ch = torch.load( model_name)
Lost_Mask_model.load_state_dict(model_ch)
Lost_Mask_model.eval()

class UpBlock(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpBlock, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)

        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2) # New!!
        self.convC = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnC = nn.BatchNorm2d(output_features)
        self.leakyreluC = nn.LeakyReLU(0.2) # New!!

        self.convD = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnD = nn.BatchNorm2d(output_features)
        self.leakyreluD = nn.LeakyReLU(0.2) # New!!

        self.convE = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnE = nn.BatchNorm2d(output_features)
        self.leakyreluE = nn.LeakyReLU(0.2) # New!!

        self.convF = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnF = nn.BatchNorm2d(output_features)
        self.leakyreluF = nn.LeakyReLU(0.2) # New!!

    def forward(self, x, concat_with):
        # print('1', x.shape, concat_with.shape)
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        # print('2', up_x.shape, concat_with.shape, '\n')
        A = self.leakyreluA(self.bnA(self.convA(torch.cat([up_x, concat_with], dim=1))))
        B = self.leakyreluB(self.bnB(self.convB(A)) +A)
        C = self.leakyreluC(self.bnC(self.convC(B)) +B)
        D = self.leakyreluD(self.bnD(self.convD(C)) +C)
        E = self.leakyreluE(self.bnE(self.convE(D)) +D)
        G = self.leakyreluF(self.bnF(self.convF(E)) +E)
        return G
#         # 3, 64, 64, 128, 256, 1024 -> densenet121
#         # 3, 96, 96, 192, 384, 2208 -> densenet161
#         # 4, 8,  16,  32,  64,   64
#***********************************************************************************************************************
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, vis,channels):
        super(Attention, self).__init__()
        self.vis = vis
        self.channels = channels*3
        self.num_attention_heads = 12
        self.attention_head_size = int(self.channels / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.channels, self.all_head_size)
        self.key = Linear(self.channels, self.all_head_size)
        self.value = Linear(self.channels, self.all_head_size)

        self.out = Linear(self.channels, self.channels)
        self.attn_dropout = Dropout(0.0) # update 0.0 to 0.01
        self.proj_dropout = Dropout(0.0) # 8-14 0.01 -> 0

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self,channels):
        super(Mlp, self).__init__()
        self.channels = channels*3
        self.hidden_channnels = channels*4
        self.fc1 = Linear(self.channels, self.hidden_channnels)
        self.fc2 = Linear(self.hidden_channnels, self.channels)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Mlp_new(nn.Module):
    def __init__(self, act_layer=nn.GELU, drop=0, channels=256):
        super().__init__()
        self.channels = channels*3
        self.out_channels = self.channels*4
        self.fc1 = nn.Linear(self.channels, self.channels*4)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.channels*4, self.channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# input B*256*16*8
# output B*
class Embeddings_single(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self,channels=256):
        super(Embeddings_single, self).__init__()
        self.channels = channels
        # self.patch_embeddings = nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0)
        if self.channels == 1024:
            self.out_channels = 1024*3
            self.HmutiW = 8*4
        if self.channels == 256:
            self.out_channels = 256*3
            self.HmutiW = 16*8
        if self.channels == 128:
            self.out_channels = 128*3
            self.HmutiW = 32*16
        if self.channels == 64:
            self.out_channels = 64*3
            self.HmutiW = 64*32
        if self.channels == 32:
            self.channels = 64
            self.out_channels = 64*3
            self.HmutiW = 128*64
        self.patch_embeddings = nn.Conv2d(self.channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        # self.patch_embeddings = nn.Conv2d(in_features=256, output_features=768, kernel_size=1, stride=1, padding=0)

        # self.patch_embeddings = Conv2d(in_channels=in_channels,         # 256
        #                                out_channels=config.hidden_size, # 768
        #                                kernel_size=patch_size,          # 1*1
        #                                stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.HmutiW, self.out_channels))
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches=256, config.hidden_size = 768))

        self.dropout = Dropout(0.1)  # 0.1

    def forward(self, x):
        # [B, C, H, W] -> [B, 256, 16, 8] 1024 8 4
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) 图像分割***
        # [B, 768, 16, 8] 768 8 4

        x = x.flatten(2) #[128, 768, 128]
        # [B, 768, 16*8]

        x = x.transpose(-1, -2)  # (B, n_patches, hidden)  图像合成（B，patch个数，隐藏层个数） #[128, 128, 768]
        # [B, 16*8, 768]

        embeddings = x + self.position_embeddings  #[128, 128, 768]
        # [B, 16*8+1, 768] B 32 768
        embeddings = self.dropout(embeddings)

        return embeddings

class Block(nn.Module):
    def __init__(self, vis, channels):
        super(Block, self).__init__()
        self.channels = channels
        self.out_channels = self.channels*3
        self.hidden_size = self.out_channels
        self.attention_norm = LayerNorm(self.out_channels, eps=1e-6)
        self.ffn_norm = LayerNorm(self.out_channels, eps=1e-6)
        self.ffn = Mlp(self.channels)
        self.attn = Attention(vis, self.channels)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder_transfromer(nn.Module):
    def __init__(self, vis=False,channels=256,num=4):
        super(Encoder_transfromer, self).__init__()
        self.vis = vis
        self.num = num
        self.channels = channels
        self.layer = nn.ModuleList()
        if self.channels == 32:
            self.out_channels=self.channels*3*2
        else:
            self.out_channels = self.channels * 3
        self.encoder_norm = LayerNorm(self.out_channels, eps=1e-6)
        self.num = num
        for i in range(self.num):
            if self.channels == 1024:
                window_size = 2  # 8*16
            if self.channels == 256:
                window_size = 4 # 8*16
            if self.channels == 128:
                window_size = 8 # 16*32
            if self.channels == 64:
                window_size = 16 # 16*32
            if self.channels == 32:
                # self.channels = 64
                window_size = 32  # 16*32
            # layer = SwinTransformerBlock(shift_size=0 if (i % 2 == 0) else window_size // 2, drop_path=0.,channels=self.channels)
            layer = Block(vis, self.channels) # Trans-Unet old_Mlp
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):

        for layer_block in self.layer:
            # hidden_states, weights = layer_block(hidden_states) # old transformer
            hidden_states = layer_block(hidden_states) # swim transformer

        # [B, 16*8, 768]
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderCup(nn.Module):
    def __init__(self,channels=256):
        super().__init__()
        self.channels = channels
        if self.channels == 1024:
            self.head_channels = 1024
            self.out_channels = 1024*3
        if self.channels == 256:
            # self.head_channels = 512 # C3 = 512
            self.head_channels = 256
            self.out_channels = 768
        if self.channels == 128:
            self.head_channels = 128
            self.out_channels = 128*3
        if self.channels == 64:
            self.head_channels = 64
            self.out_channels = 64*3
        if self.channels == 32:
            self.head_channels = 64
            self.out_channels = 64*3

        self.conv_more = Conv2dReLU(
            self.out_channels,
            self.head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, hidden_states):
        # [128, 128, 768]
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        if self.channels == 1024:
            h, w = 4, 8
        if self.channels == 256:
            h, w = 8, 16
        if self.channels == 128:
            h, w = 16, 32
        if self.channels == 64:
            h, w = 32, 64
        if self.channels == 32:
            h, w = 64, 128
        x = hidden_states.permute(0, 2, 1)
        # [128, 768, 128]

        x = x.contiguous().view(B, hidden, h, w)
        # [128, 768, 8, 16]

        x = self.conv_more(x)
        return x

class Transformer(nn.Module):
    def __init__(self, channels=256, num=4 ):
        super(Transformer, self).__init__()
        # [B, 16*8, 768]
        self.channels = channels
        self.num = num
        self.embeddings = Embeddings_single(channels=self.channels) #size=256*128
        self.encoder = Encoder_transfromer(channels=self.channels,num=self.num)
        self.encoder_cup = DecoderCup(channels=self.channels)
        
    def forward(self, input_ids):
        # embedding_output需要进行多一步embedding操作
        # input_ids = 16*8
        # input_ids2 = 8*4

        # [B, 256, 16, 8] -> [B, 16*8, 768] position_embedd
        embedding_output = self.embeddings(input_ids) # 输出最小特征经过嵌入后的特征图，高分辨率的特征图

        # [B, 8*16, 768] -> [B, 8*16, 768]
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)

        # [B, 8*16, 768] -> [B, 768, 8, 16]
        encoded_transfromer = self.encoder_cup(encoded)
        return encoded_transfromer

#***********************************************************************************************************************

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, shift_size=0, drop_path=0.,channels=256): # drop_path=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        super().__init__()
        self.channels = channels
        self.dim = self.channels * 3  # 768 #输入的通道数
        if channels == 1024:
            self.input_resolution = (4, 8) # 存在patchembedding降维操作
            self.window_size = 2  # 窗口的尺寸 4
        if channels == 256:
            self.input_resolution = (8, 16) # 存在patchembedding降维操作
            self.window_size = 4  # 窗口的尺寸 4
        if channels == 128:
            self.input_resolution = (16, 32) # 存在patchembedding降维操作
            self.window_size = 8  # 窗口的尺寸 4
        if channels == 64:
            self.input_resolution = (32, 64) # 存在patchembedding降维操作
            self.window_size = 16  # 窗口的尺寸 4
        if channels == 32:
            # self.channels = 64
            self.input_resolution = (64, 128) # 存在patchembedding降维操作
            self.window_size = 32  # 窗口的尺寸 4
            self.dim = 192  # 768 #输入的通道数
        self.num_heads = 12 # 注意力的head数量
        self.shift_size = shift_size #SW-MSA滑移窗口每次的移动距离
        self.mlp_ratio = 4. # MLP的隐藏层维度与patch embedding 维度的比例

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"  # 2<4


        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=12,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 =nn.LayerNorm(self.dim)
        # mlp_hidden_dim = int(self.dim * 4.)
        if self.channels == 32:
            self.mlp = Mlp(drop=0.01,channels=self.channels*2)
        else:
            self.mlp = Mlp(drop=0.01, channels=self.channels)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), #
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x): # x = [B, 8*16, 768]
        # 获取宽度高度方向分别分成了几个patch

        if self.channels == 1024:
            H, W = 4, 8
        if self.channels == 256:
            H, W = 8, 16
        if self.channels == 128:
            H, W = 16, 32
        if self.channels == 64:
            H, W = 32, 64
        if self.channels == 32:
            H, W = 64, 128
        B, L, C = x.shape  # input = [B ，H*W，C=768]
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # [B ，H*W，C] -> [B ，H，W，C]
        x = x.view(B, H, W, C) # x = [B, 8, 16, 768]

        # cyclic shift  使用图像循环移动代替窗口移动，极大地减轻了算法的工程量！！！
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size*2), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows 进行窗口的分割，输入为移动后的X，输出为（nW*B, window_size, window_size, C）
        x_windows = window_partition(shifted_x, self.window_size) #[8*B, 4, 4, 768]
        # 调整窗口的形状为（nW*B, window_size*window_size, C）
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  #[8*B, 4*4, 768]

        # W-MSA/SW-MSA 对于每个子窗口计算局部注意力，尺寸为（nW*B, window_size*window_size, C）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)    # [8*B, 4*4, 768]
        # 
        # merge windows  # [8*B, 4, 4, 768]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # 调整窗口的形状为（nW*B, window_size，window_size, C）

        # => [B, 8, 16, 768]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # 将注意力的Batch还原为与图像batch数相同的尺寸（B，H，W，C）

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size*2), dims=(1, 2))
        else:
            x = shifted_x

        # => [B, 8*16, 768]
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
#***********************************************************************************************************************

class Decoder(nn.Module):
    def __init__(self, num_features=1024, feature_base=256,  decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.inconv = nn.Conv2d(num_features+64+64+64, features, kernel_size=1, stride=1, padding=1)  # 1024+64+64+64 -> 512 no Densenet

        self.up1 = UpBlock(skip_input=features//1  + 64+64+64+ feature_base, output_features=features // 2)    #512+64+64+64+512 -> 256 no Transformer

        self.up2 = UpBlock(skip_input=features//2  +32+32+32+ feature_base//2 , output_features=features // 4)  #256+32+32+32+128 -> 128 no Dense

        self.up3 = UpBlock(skip_input=features//4  +16+16+16+feature_base//4, output_features=features // 8)     # 128+16+16+16+64 -> 64 no Dense

        self.up4 = UpBlock(skip_input=features//8  +8+8+8+feature_base//4, output_features=features // 16)          # 64+8+8+8+64 -> 32 no Dense

        self.up5 = UpBlock(skip_input=features//16 +4+4+4+ 3, output_features=features // 32)         # 32+4+4+4+3 no Dense

        self.outconv = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)
	
        # 是否使用 Transformer Blocks
        # self.transformer_c4 = Transformer(channels=1024,num=2)
        # self.transformer_c3 = Transformer(channels=256,num=6)
        # self.transformer_c2 = Transformer(channels=128,num=6)
        # self.transformer_c1 = Transformer(channels=64,num=4)
        # self.transformer_c0 = Transformer(channels=32,num=2)

    def forward(self, features_color, features_depth, features_normal, features_outline):

        # 3, 64, 64, 128, 256, 1024 -> densenet121
        # 3, 96, 96, 192, 384, 2208 -> densenet161
        # 4, 8,  16,  32,  64,   64
        c_in, c_block0, c_block1, c_block2, c_block3, c_block4 = \
            features_color[0], features_color[3], features_color[4], \
            features_color[6], features_color[8], features_color[11] 

        d_in, d_block0, d_block1, d_block2, d_block3, d_block4 = \
            features_depth[0], features_depth[1], features_depth[2], \
            features_depth[3], features_depth[4], features_depth[5]

        n_in, n_block0, n_block1, n_block2, n_block3, n_block4 = \
            features_normal[0], features_normal[1], features_normal[2], \
            features_normal[3], features_normal[4], features_normal[5]

        o_in, o_block0, o_block1, o_block2, o_block3, o_block4 = \
            features_outline[0], features_outline[1], features_outline[2], \
            features_outline[3], features_outline[4], features_outline[5]

        # 是否使用 Transformer Blocks
        # c_block4 = self.transformer_c4(c_block4)
        # c_block3 = self.transformer_c3(c_block3)
        # c_block2 = self.transformer_c2(c_block2)
        # c_block1 = self.transformer_c1(c_block1)
        # c_block0 = self.transformer_c0(c_block0)

        x_d0 = self.inconv(torch.cat([c_block4, d_block4, n_block4, o_block4], dim=1))      #     1024  * 512 -> 1/32 1024+512+512+64

        x_d1 = self.up1(x_d0, torch.cat([c_block3, d_block3, n_block3, o_block3], dim=1))  # (512+256) * 256 -> 1/16

        x_d2 = self.up2(x_d1, torch.cat([c_block2, d_block2, n_block2, o_block2], dim=1))  # (128+128) * 128 -> 1/8

        x_d3 = self.up3(x_d2, torch.cat([c_block1, d_block1, n_block1, o_block1], dim=1))  # (64 + 64) *  64 -> 1/4

        x_d4 = self.up4(x_d3, torch.cat([c_block0, d_block0, n_block0, o_block0], dim=1))  # (32 + 64) *  32 -> 1/2

        x_d5 = self.up5(x_d4, torch.cat([c_in, d_in, n_in, o_in], dim=1))  # (32 + 64) *  32 -> 1/2
        return self.outconv(x_d5)

class Encoder(nn.Module):
    def __init__(self, densenet='121'):
        super(Encoder, self).__init__()       
        import torchvision.models as models
        if densenet == '161':
            self.original_model = models.densenet161(pretrained=True)
            print('Use Pretrain Densenet161 Model.')
        else:
            self.original_model = models.densenet121(pretrained=True, memory_efficient=False)
            print('Use Pretrain Densenet121 Model.')

        for k, v in self.original_model.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        features = [x]
        """
        Block  0 ->    3 | 1/1 
        Block  1 ->   64 | 1/2
        Block  2 ->   64 | 1/2
        Block  3 ->   64 | 1/2   **
        Block  4 ->   64 | 1/4   **
        Block  5 ->  256 | 1/4
        Block  6 ->  128 | 1/8   **
        Block  7 ->  512 | 1/8
        Block  8 ->  256 | 1/16  **
        Block  9 -> 1024 | 1/16
        Block 10 ->  512 | 1/32
        Block 11 -> 1024 | 1/32  **
        Block 12 -> 1024 | 1/32
        """
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features

class DownBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(DownBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)

        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2)  #New!!

        self.convC = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnC = nn.BatchNorm2d(output_features)
        self.leakyreluC = nn.LeakyReLU(0.2)  #New!!

        self.convD = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnD = nn.BatchNorm2d(output_features)
        self.leakyreluD = nn.LeakyReLU(0.2)  #New!!

        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, ceil_mode=False)

    def forward(self, x):
        x = self.pool(x)
        A = self.leakyreluA(self.bnA(self.convA(x)))
        B = self.leakyreluB(self.bnB(self.convB(A)) + A)
        C = self.leakyreluC(self.bnC(self.convC(B)) + B)
        D = self.leakyreluD(self.bnD(self.convD(C)) + C)
        return D

class DownBlock_1in(nn.Sequential):
    def __init__(self, input, output_features):
        super(DownBlock_1in, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)

        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2)  # New!!

        self.convC = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnC = nn.BatchNorm2d(output_features)
        self.leakyreluC = nn.LeakyReLU(0.2)  # New!!

        self.convD = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnD = nn.BatchNorm2d(output_features)
        self.leakyreluD = nn.LeakyReLU(0.2)  # New!!

        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, ceil_mode=False)

        self.res = nn.Conv2d(input, output_features, kernel_size=1, stride=2, padding=0)

    def forward(self, x, sample_down=True):
        if sample_down:
            LDM = self.res(x)
            x = self.pool(x)
            A = self.leakyreluA(self.bnA(self.convA(x)) + LDM)

        else:
            A = self.leakyreluA(self.bnA(self.convA(x)) + x)

        B = self.leakyreluB(self.bnB(self.convB(A)) + A)
        C = self.leakyreluC(self.bnC(self.convC(B)) + B)
        D = self.leakyreluD(self.bnD(self.convD(C)) + C)

        return D

class DownBlock_2in(nn.Sequential):
    def __init__(self, input, output_features):
        super(DownBlock_2in, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2)  # New!!
        self.pool = nn.AvgPool2d(kernel_size=2, padding=0, ceil_mode=False)

        self.res = nn.Conv2d(input, output_features, kernel_size=1, stride=2, padding=0)

    def forward(self, DM, LDM, sample_down=True):
        # if sample_down is True:
        #     LDM = self.res(LDM)
        #     x = self.pool(x)
        A = self.leakyreluA(self.bnA(self.convA(LDM)))
        B = self.bnB(self.convB(A))
        return  self.leakyreluB( B + DM )
        # return self.leakyreluB(self.convB(self.leakyreluA(self.convA(self.pool(x)))) + self.res(x2))

class InBlock(nn.Sequential):
    def __init__(self, input, output_features):
        super(InBlock, self).__init__()
        self.convA = nn.Conv2d(input, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(output_features)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(output_features)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x):
        A = self.leakyreluA(self.bnA(self.convA(x)))
        B = self.leakyreluB(self.bnB(self.convB(A)) +A)
        return B

class Encoder_Depth(nn.Module):
    def __init__(self):
        super(Encoder_Depth, self).__init__()

        self.In = InBlock(1, 4)
        self.d0 = DownBlock(4, 8)
        self.d1 = DownBlock(8, 16)
        self.d2 = DownBlock(16, 32)
        self.d3 = DownBlock(32, 64)
        self.d4 = DownBlock(64, 64)

    def forward(self, features):
        x_in = self.In(features)  # 1*4   -> 1/1
        x_d0 = self.d0(x_in)      # 4*8   -> 1/2
        x_d1 = self.d1(x_d0)      # 8*16  -> 1/4
        x_d2 = self.d2(x_d1)      # 16*32 -> 1/8
        x_d3 = self.d3(x_d2)      # 32*64 -> 1/16
        x_d4 = self.d4(x_d3)      # 64*64 -> 1/32
        return [x_in, x_d0, x_d1, x_d2, x_d3, x_d4]

class Encoder_Depth_depth(nn.Module):
    def __init__(self):
        super(Encoder_Depth_depth, self).__init__()
        self.In = InBlock(1, 4)
        self.d0 = DownBlock(4, 8)
        self.d1 = DownBlock(8, 16)
        self.d2 = DownBlock(16, 32)
        self.d3 = DownBlock(32, 64)
        self.d4 = DownBlock(64, 64)

    def forward(self, features):
        x_in = self.In(features)  # 1*4   -> 1/1
        x_d0 = self.d0(x_in)      # 4*8   -> 1/2
        x_d1 = self.d1(x_d0)      # 8*16  -> 1/4
        x_d2 = self.d2(x_d1)      # 16*32 -> 1/8
        x_d3 = self.d3(x_d2)      # 32*64 -> 1/16
        x_d4 = self.d4(x_d3)      # 64*64 -> 1/32
        return [x_in, x_d0, x_d1, x_d2, x_d3, x_d4]

class Mask_Net(nn.Module):
    def __init__(self):
        super(Mask_Net, self).__init__()
        self.In = InBlock(1, 4)
        self.d0 = DownBlock(4, 8)
        self.d1 = DownBlock(8, 16)
        self.d2 = DownBlock(16, 32)
        self.d3 = DownBlock(32, 64)
        self.d4 = DownBlock(64, 64)

    def forward(self, features):
        x_in = self.In(features.float())  # 1*4   -> 1/1
        x_d0 = self.d0(x_in)      # 4*8   -> 1/2
        x_d1 = self.d1(x_d0)      # 8*16  -> 1/4
        x_d2 = self.d2(x_d1)      # 16*32 -> 1/8
        x_d3 = self.d3(x_d2)      # 32*64 -> 1/16
        x_d4 = self.d4(x_d3)      # 64*64 -> 1/32
        return [x_in, x_d0, x_d1, x_d2, x_d3, x_d4]

class Residual_Mask_Net(nn.Module):
    def __init__(self):
        super(Residual_Mask_Net, self).__init__()

        self.In = InBlock(1, 4)
        
        self.d0 = DownBlock_2in(4, 4)
        self.d0_2 = DownBlock_1in(4, 8)
        
        self.d1 = DownBlock_1in(8, 8)
        self.d1_2 = DownBlock_1in(8, 16)
        
        self.d2 = DownBlock_1in(16, 16)
        self.d2_2 = DownBlock_1in(16, 32)
        
        self.d3 = DownBlock_1in(32, 32)
        self.d3_2 = DownBlock_1in(32, 64)
        
        self.d4 = DownBlock_1in(64, 64)
        self.d4_2 = DownBlock_1in(64, 64)

    def forward(self, Mask_Depth, LostMasked_Depth):
        MD_in = self.In(Mask_Depth.float())
        LMD_in = self.In(LostMasked_Depth.float())
        # x_in = self.In(features)         # 1*4   -> 1/1

        MD_d0 = self.d0(MD_in, LMD_in , sample_down=False)      # 4->4
        MD_d0 = self.d0_2(MD_d0, sample_down=True)              # 4->8
        
        MD_d1 = self.d1(MD_d0, sample_down=False)      # 8*16  -> 1/4
        MD_d1 = self.d1_2(MD_d1, sample_down=True)      # 8*16  -> 1/4
        
        MD_d2 = self.d2(MD_d1, sample_down=False)      # 16*32 -> 1/8
        MD_d2 = self.d2_2(MD_d2, sample_down=True)      # 16*32 -> 1/8
        
        MD_d3 = self.d3(MD_d2, sample_down=False)      # 32*64 -> 1/16
        MD_d3 = self.d3_2(MD_d3, sample_down=True)      # 32*64 -> 1/16
        
        MD_d4 = self.d4(MD_d3, sample_down=False)      # 64*64 -> 1/32
        MD_d4 = self.d4_2(MD_d4, sample_down=True)      # 64*64 -> 1/32

        MD_new_in = torch.cat([MD_in, LMD_in], dim= 1)
        return [MD_new_in, MD_d0, MD_d1, MD_d2, MD_d3, MD_d4]

class Model(nn.Module):
    def __init__(self, pretrain_model='121',is_real=False):
        super(Model, self).__init__()
        if pretrain_model == '121':
            self.encoder = Encoder(densenet='121')
            self.decoder = Decoder(num_features=1024, feature_base=256)
        else:
            self.encoder = Encoder(densenet='161')
            self.decoder = Decoder(num_features=2208, feature_base=384)
            
        # 使用 DenseNet
        # self.encoder_normal = Encoder(densenet='121')
        # self.encoder_outline = Encoder(densenet='121')
        # self.Res_Mask = Residual_Mask_Net()
        # self.In_normal = InBlock(1, 3)
        # self.In_outline = InBlock(1, 3)

        # 不使用 DenseNet
        self.encoder_normal = Encoder_Depth()
        self.encoder_outline = Encoder_Depth()
        # self.Res_Mask = Residual_Mask_Net()
        self.Res_Mask = Mask_Net()


    def forward(self, x ,is_real=False):
        color = x[:, :3, :, :]
        depth_orig = x[:, 3:4, :, :]
        normal = x[:, 4:5, :, :]
        outline = x[:, 5:6, :, :]
        mask = x[:, 6:7, :, :]  # 0.0039
        another_mask = x[:, 7:8, :, :] # 0.0039

        orig_mask = torch.from_numpy(np.where(depth_orig.cpu() > 0, 1, 0)).cuda()
        invalid_mask = (1-orig_mask) + (mask) * 255
        one_invalid_mask = torch.from_numpy(np.where(invalid_mask.cpu() > 0, 1, 0)).cuda()
        # plt.imshow(Lost_Mask.cpu()[0, 0, :, :])
        # plt.show()
        if is_real:
            gt_depth = x[:, 8:9, :, :]
            clue_depth= x[:,9:10,:,:]
            valid_mask = clue_depth* mask * 255

        else:
            gt_depth = x[:, 8:9, :, :]
            valid_mask =  mask * 255

        pre = self.decoder(self.encoder(color), self.Res_Mask(valid_mask), self.encoder_normal(normal),  self.encoder_outline(outline) )

        return pre, valid_mask



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#作用是将若干个分割好的小窗口，还原成一个图像batch。
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# class Mydataset(Dataset):
#     def __init__(self, df_data, data_dir='./',transform=None):
#         super().__init__()
#         self.df=df_data.values
#         self.data_dir=data_dir
#         self.transform=transform
# 
#     def __len__(self):
#         return len(self.df)
# 
#     def __getitem__(self,idex):
#         color_name, another_mask_name, label_name=self.df[idex]
# 
#         color_path=os.path.join(self.data_dir,color_name)
#         another_mask_path=os.path.join(self.data_dir,another_mask_name)
#         label_path=os.path.join(self.data_dir,label_name)
# 
#         color_image=cv2.imread(color_path)
#         another_mask_image = 1-cv2.imread(another_mask_path) #0011100
#         label_image=1-cv2.imread(label_path) #0010100
# 
#         if self.transform is not None:
#             color_image = self.transform(color_image)
#             another_mask_image = self.transform(another_mask_image)
#             label_image = self.transform(label_image)
# 
#         another_mask_image = another_mask_image[0, :, :]
#         label_image = label_image[0, :, :]
# 
#         # plt.subplot(221)
#         # plt.imshow(another_mask_image)
#         # plt.subplot(222)
#         # plt.imshow(color_image[0,:,:])
#         # # another_mask_image = np.where(another_mask_image>0,1,0)
#         # # label_image = np.where(label_image>0,1,0)
#         # plt.subplot(223)
#         # plt.imshow(another_mask_image*color_image[0,:,:])
#         # plt.subplot(224)
#         # plt.imshow(label_image)
#         # plt.show()
#         color_masked = another_mask_image * color_image
#         another_mask_image = torch.from_numpy(np.expand_dims(another_mask_image, axis=0))
#         label_image = torch.from_numpy(np.expand_dims(label_image, axis=0))
# 
#         return color_masked, another_mask_image, label_image
# 
