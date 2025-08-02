import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import numpy as np

class MedSAM_Lite(nn.Module):
    def __init__(self, model_checkpoint=None, device='cpu'):
        super().__init__()
        self.device = device

        # Load image encoder
        self.image_encoder = TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )

        # Load prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16,
        )

        # Load mask decoder
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if model_checkpoint:
            state_dict = torch.load(model_checkpoint, map_location=self.device)
            print("Checkpoint keys:", state_dict.keys())  # Debug print
            self.load_weights(state_dict)

        self.to(self.device)

    def load_weights(self, state_dict):
        if 'image_encoder' in state_dict:
            print("Loading modular components...")
            self.image_encoder.load_state_dict(state_dict['image_encoder'])
            self.prompt_encoder.load_state_dict(state_dict['prompt_encoder'])
            self.mask_decoder.load_state_dict(state_dict['mask_decoder'])
        else:
            print("Loading full model state_dict...")
            self.load_state_dict(state_dict)

    def forward(self, image, boxes):
        image = image.to(self.device)

        # Image encoding
        image_embedding = self.image_encoder(image)

        # Convert boxes to prompt inputs
        box_tensors = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
        box_tensors = box_tensors[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_tensors,
            masks=None,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks

