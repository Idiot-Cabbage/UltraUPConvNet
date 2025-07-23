from matplotlib.sankey import UP
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_base, convnext_tiny
#from .decoder_UPerhead import UPerHead

class OmniVisionConvNeXt(nn.Module):
    def __init__(self, args, prompt=False):
        super().__init__()
        self.prompt = prompt
        # ConvNeXt backbone
        if args.use_pretrained_model:
            #self.backbone = convnext_base(pretrained=False, num_classes=1000)
            self.backbone = convnext_tiny(pretrained=False, num_classes=1000)
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            #self.backbone = convnext_base(pretrained=False, num_classes=1000)
            self.backbone = convnext_tiny(pretrained=False, num_classes=1000)
            print("No pretrained weights provided for ConvNeXt.")
        
        embed_dim = 768  # ConvNeXt classification 最后一层维度
        self.dec_prompt_mlp = nn.Linear(15, embed_dim) if prompt else None
        self.prompt_proj_layers = nn.ModuleList([
            nn.Linear(15, 96),
            nn.Linear(15, 192),
            nn.Linear(15, 384),
            nn.Linear(15, 768),
        ]) if prompt else None
        # self.prompt_layers = nn.ModuleList([
        #     nn.Linear(15, embed_dim),
        #     nn.Linear(15, embed_dim)
        # ]) if prompt else None

        # Segmentation head
        from .decoder_UPerhead import UPerNet
        self.seg_head = UPerNet()
        
        # Classification head
        self.cls_head_2cls = nn.Linear(embed_dim, 2)
        self.cls_head_4cls = nn.Linear(embed_dim, 4)

    def forward(self, x):
        if self.prompt:
            image = x[0]  # (B, 1, H, W, C)
            #print(image.shape)
            position_prompt = x[1]
            task_prompt = x[2]
            type_prompt = x[3]
            nature_prompt = x[4]
            B = image.shape[0]
            image = image.squeeze(1).permute(0, 3, 1, 2)  # (B, C, H, W)
            prompt_input = torch.cat([position_prompt, task_prompt, type_prompt, nature_prompt], dim=1)  # (B, 15)
        else:
            image = x.squeeze(1).permute(0, 3, 1, 2)
            B = image.shape[0]
                
        feats_cls, feats_seg = self.backbone.forward(image)

        if self.prompt:
            prompt_cls = self.dec_prompt_mlp(prompt_input).view(B, -1, 1, 1)  # (B, C)
            feats_cls = feats_cls.view(B, -1, 1, 1)
            feats_cls = feats_cls + prompt_cls
            # print("feats_cls shape after prompt:", feats_cls.shape)  # torch.Size([8, 768, 7, 7])
            feats_seg_with_prompt = []
            for i in range(len(feats_seg)):
                prompt_seg = self.prompt_proj_layers[i](prompt_input).view(B, -1, 1, 1)  # (B, C_i, 1, 1)
                print(feats_seg[i].shape, prompt_seg.shape, (feats_seg[i] + prompt_seg).shape)
                feats_seg_with_prompt.append(feats_seg[i] + prompt_seg)
        
            feats_seg = feats_seg_with_prompt
                 
        # Segmentation
        seg_out = self.seg_head(image, feats_seg)  # (B, 2, H, W)

        # Classification
        #pooled_feats = F.adaptive_avg_pool2d(feats_cls, 1).view(feats_cls.size(0), -1)  # (B, C)
        pooled_feats = feats_cls.view(B, -1)
        cls_out_2cls = self.cls_head_2cls(pooled_feats)
        cls_out_2cls = F.softmax(cls_out_2cls, dim=1)
        
        cls_out_4cls = self.cls_head_4cls(pooled_feats)
        cls_out_4cls = F.softmax(cls_out_4cls, dim=1)
        
        return (seg_out, cls_out_2cls, cls_out_4cls)
