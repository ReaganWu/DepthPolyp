import torch
import torch.nn as nn

from .modules.HF_Decoder import HiF_Decoder
from .modules.MiT_Encoder import MixVisionTransformer
from .modules.Seg_Head import SegmentationHead

class DepthPolyp(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2, # 1 for seg, 1 for depth
        encoder_name: str = 'b0',
        decoder_channels: int = 256,
        activation: str = None,
        upsampling: int = 4,
    ):
        super().__init__()

        # Encoder configurations
        encoder_configs = {
            'b0': {
                'embed_dims': [32, 64, 160, 256],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [2, 2, 2, 2],
                'sr_ratios': [8, 4, 2, 1],
            },
            'b1': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [2, 2, 2, 2],
                'sr_ratios': [8, 4, 2, 1],
            },
            'b2': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [3, 4, 6, 3],
                'sr_ratios': [8, 4, 2, 1],
            },
            'b3': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [3, 4, 18, 3],
                'sr_ratios': [8, 4, 2, 1],
            },
            'b4': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [3, 8, 27, 3],
                'sr_ratios': [8, 4, 2, 1],
            },
            'b5': {
                'embed_dims': [64, 128, 320, 512],
                'num_heads': [1, 2, 5, 8],
                'mlp_ratios': [4, 4, 4, 4],
                'depths': [3, 6, 40, 3],
                'sr_ratios': [8, 4, 2, 1],
            },
        }

        if encoder_name not in encoder_configs:
            raise ValueError(f"encoder_name should be one of {list(encoder_configs.keys())}, got {encoder_name}")

        config = encoder_configs[encoder_name]

        # Build encoder
        self.encoder = MixVisionTransformer(
            in_chans=in_channels,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            mlp_ratios=config['mlp_ratios'],
            qkv_bias=True,
            depths=config['depths'],
            sr_ratios=config['sr_ratios'],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


        self.decoder = HiF_Decoder(
            encoder_channels=config['embed_dims'],
            decoder_channels=decoder_channels,
        )

        # Build segmentation head (nn.Sequential style)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels//4,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.name = f"segformer-{encoder_name}"

    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        # Encoder - returns features at [H/4, H/8, H/16, H/32]
        encoder_features = self.encoder(x)
        
        # Decoder - returns features at H/4

        fpn_features = self.decoder(encoder_features)
        decoder_output = fpn_features
        # print(f"Decoder output shape: {decoder_output.shape}")

        # Segmentation head - upsample to original size
        masks = self.segmentation_head(decoder_output)
        pred_seg = torch.sigmoid(masks[:, 0:1, :, :])   # segmentation 通道
        pred_depth = torch.sigmoid(masks[:, 1:2, :, :])                # depth 通道，通常是回归，不做激活

        return pred_seg, pred_depth

    @torch.no_grad()
    def predict(self, x):
        """Inference method"""
        if self.training:
            self.eval()
        return self(x)

    def load_pretrained(self, checkpoint_path, strict=True):
        """Load pretrained weights
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove module. prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict, strict=strict)
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")


def build_depthpolyp(
    encoder_name='b0',
    in_channels=3,
    num_classes=2,
    decoder_channels=256,
    activation=None,
):
    """
    Create a DepthPolyp model
    
    Args:
        encoder_name: Encoder variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
        in_channels: Number of input channels
        num_classes: Number of output classes
        decoder_channels: Number of channels in decoder
        activation: Output activation ('sigmoid', 'softmax', or None)
    
    Returns:
        DepthPolyp model
    
    Example:
        >>> model = build_depthpolyp('b2', num_classes=21, activation='softmax')
        >>> print(model)
    """

    model = DepthPolyp(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_name=encoder_name,
        decoder_channels=decoder_channels,
        activation=activation,
    )
    return model

if __name__ == '__main__':
    print("="*60)
    print("Loading Model .....")
    model = build_depthpolyp(
        encoder_name='b0',
        in_channels=3, # Input channels
        num_classes=2, # Total 2. 1 for seg, 1 for depth
        decoder_channels=256,
        activation='sigmoid',
    )
    print("="*60)
    print("Validating Model .....")
    print("Check the Param and Complexity(GMACs)")
    import ptflops
    macs, params = ptflops.get_model_complexity_info(
        model, (3, 224, 224), as_strings=True,
        print_per_layer_stat=False, verbose=False
    )
    print(f"   MACs: {macs}, Params: {params}")
    # output is MACs: 862.17 MMac, Params: 3.57 M
    print("="*60)
    print("Check the output .....")
    dummy_input = torch.randn(1, 3, 224, 224) # B, C, H, W, single RGB image
    output_seg, output_depth = model(dummy_input)
    print("input_shape is:", dummy_input.shape)
    print("output_seg shape is:", output_seg.shape)
    print("output_depth shape is:", output_depth.shape)

