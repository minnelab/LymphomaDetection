from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.layers import Conv
from monai.networks.nets import SwinUNETR
import torch

class SwinAutoEnc(SwinTransformer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        
        hidden_size = self.embed_dim*16
        deconv_chns = 16

        self.out_channels = out_channels

        spatial_dims = 3
        if "spatial_dims" in kwargs:
            spatial_dims = kwargs["spatial_dims"]

        conv_trans = Conv[Conv.CONVTRANS, spatial_dims]
        up_kernel_size = [8,8,8]

        self.conv3d_transpose = conv_trans(hidden_size, deconv_chns, kernel_size=up_kernel_size, stride=up_kernel_size)
        self.conv3d_transpose_1 = conv_trans(
                    in_channels=deconv_chns, out_channels=self.out_channels, kernel_size=[4,4,4], stride=[4,4,4]
                )
    
    def forward(self, x):
        x0_out, x1_out, x2_out, x3_out, x4_out = super().forward(x)
        x = self.conv3d_transpose(x4_out)
        x = self.conv3d_transpose_1(x)
        return x
    
    
class RetinaSwinUNETR(SwinUNETR):
    def __init__(self, out_feature_channels, **kwargs):
        super().__init__(**kwargs)
        
        self.out_channels: int = out_feature_channels

        self.feature_size = 24
        if "feature_size" in kwargs:
            self.feature_size = kwargs["feature_size"]
            
        spatial_dims = 3
        if "spatial_dims" in kwargs:
            spatial_dims = kwargs["spatial_dims"]
        
        self.spatial_dims = spatial_dims
        
    
        
        conv = Conv[Conv.CONV, spatial_dims]
        
        #self.conv3d_4 = conv( self.feature_size*16, self.out_channels, kernel_size=3, stride=1, padding=1)
        #self.conv3d_3 = conv( self.feature_size*8, self.out_channels, kernel_size=3, stride=1, padding=1)
        #self.conv3d_2 = conv( self.feature_size*4, self.out_channels, kernel_size=3, stride=1, padding=1)
        #self.conv3d_1 = conv( self.feature_size*2, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3d_0 = conv( self.feature_size, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        
        #dec4_out = self.conv3d_4(dec4)
        #dec3_out = self.conv3d_3(dec3)
        #dec2_out = self.conv3d_2(dec2)
        #dec1_out = self.conv3d_1(dec1)
        dec0_out = self.conv3d_0(dec0)
        
        features = [
            #dec4_out, dec3_out, dec2_out, dec1_out,
             dec0_out]
        
        return logits, features 
        