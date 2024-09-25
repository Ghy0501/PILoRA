from timm import create_model
from functools import reduce
from operator import mul
import math
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np



class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_qs,
        linear_b_qs,
        linear_a_vs,
        linear_b_vs,
    ):
        super().__init__()
        self.qkv = qkv
        for i in range(len(linear_a_qs)):
            setattr(self, f'linear_a_q_{i}', linear_a_qs[i])
            setattr(self, f'linear_b_q_{i}', linear_b_qs[i])
            setattr(self, f'linear_a_v_{i}', linear_a_vs[i])
            setattr(self, f'linear_b_v_{i}', linear_b_vs[i])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.lora_id = 0
    
    def change_lora(self, num):
        self.lora_id = num

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        linear_a_q = getattr(self, f'linear_a_q_{self.lora_id}')
        linear_b_q = getattr(self, f'linear_b_q_{self.lora_id}')
        linear_a_v = getattr(self, f'linear_a_v_{self.lora_id}')
        linear_b_v = getattr(self, f'linear_b_v_{self.lora_id}')
        new_q = linear_b_q(linear_a_q(x))
        new_v = linear_b_v(linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class VLT(nn.Module):
    def __init__(self, 
                 modelname: str, 
                 num_classes: int, 
                 pretrained: bool = True,
                 r: int = 4,
                 lora_layer=None,
                 return_feature = True,
                 g_model = False):
        super().__init__()
        self.encoder = create_model(modelname, num_classes=num_classes, pretrained=pretrained)
        self.num_classes = num_classes
        self.return_feature = return_feature
        self.feat_dim = 768

        # seed = 2023
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        self.centers = nn.ParameterList(
            [nn.Parameter(0.1*torch.randn(self.feat_dim, 1)) for i in range(self.num_classes)])
        # for center in self.centers:
        #     center.data.fill_(1.0)
        # freeze
        for n, p in self.encoder.named_parameters():
            if 'head' not in n:
                p.requires_grad = False
        
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(self.encoder.blocks)))
        
        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        if g_model == True:
            num = 1
        else:
            num = 10

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(self.encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_qs = []
            w_b_linear_qs = []
            w_a_linear_vs = []
            w_b_linear_vs = []

            for i in range(num):
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)

                if g_model == True:
                    nn.init.zeros_(w_a_linear_q.weight)
                    nn.init.zeros_(w_a_linear_v.weight)
                else:
                    nn.init.kaiming_uniform_(w_a_linear_q.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(w_a_linear_v.weight, a=math.sqrt(5))
                nn.init.zeros_(w_b_linear_q.weight)
                nn.init.zeros_(w_b_linear_v.weight)

                w_a_linear_qs.append(w_a_linear_q)
                w_b_linear_qs.append(w_b_linear_q)
                w_a_linear_vs.append(w_a_linear_v)
                w_b_linear_vs.append(w_b_linear_v)
        
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_qs,
                w_b_linear_qs,
                w_a_linear_vs,
                w_b_linear_vs,
                )

        # self.reset_parameters()
        # self.lora_vit = self.encoder
    def centers_initial(self, current_tasks):
        current_task = [i for i in range(current_tasks[0], current_tasks[1])]
        no_grad_idx = [i for i in range(self.num_classes) if i not in current_task]
        for i in no_grad_idx:
            self.centers[i].requires_grad = False
        for i in current_task:
            self.centers[i].requires_grad = True
    
    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")
    
    def switch_lora(self, idx:int):
        for t_layer_i, blk in enumerate(self.encoder.blocks):
            blk.attn.qkv.lora_id = idx

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    
    def compute_distance(self, x):
        centers_list = [i for i in self.centers]
        centers = torch.cat(centers_list, dim=1)
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (centers))
        dist = features_square + centers_square - features_into_centers
        dist = dist / float(x.shape[1])
        dist = torch.sqrt(dist)
        
        return dist, centers
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward_features(self, x, mode):
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        # x = self.encoder.norm_pre(x)
        # x = self.encoder.blocks(x)
        for i, blk in enumerate(self.encoder.blocks):
            if i == 0:
                x = blk(x)
                feat_layer4 = x[:, 0:1, :]
                feat_layer4 = feat_layer4.squeeze(dim=1)
            else:
                x = blk(x)
        x = self.encoder.norm(x)
        return x, feat_layer4

    def forward(self, x, mode='train'):
        x, feat_layer4 = self.forward_features(x, mode)
        feature = self.encoder.forward_head(x, pre_logits=True)
        # feature = self.fc(feature)
        x = self.encoder.forward_head(x)
        # return self.lora_vit(x)
        if self.return_feature:
            dist, centers = self.compute_distance(feature)
            return x, feature, dist, centers, feat_layer4
        else:
            return x