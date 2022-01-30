import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1 = nn.Conv2d(self.in_features, self.out_features, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(self.out_features)
        self.conv2 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(self.out_features)
            
    def forward(self, x):
        identity=x
        
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        out += identity
        out = F.relu(out)
        
        return out

class FeatureExtraction(nn.Module):
    def __init__(self, in_features:int, out_features:int, groups:int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1 = nn.Conv2d(self.in_features, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.layer = self._make_layer(self.out_features, blocks=groups)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(self.out_features)
        
    def _make_layer(self, out_features:int, blocks:int) -> nn.Sequential():
        layers = []
        for i in range(blocks):
            layers.append(ResBlock(out_features, out_features))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.layer(x)
        x = self.conv2(x)
        return x

class ThreeDConv(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        #self.conv3 = nn.Conv3d(channels,channels, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm3d(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        return x

class GCnet(nn.Module):
    def __init__(self, maxdisp, name="GCnet"):
        super().__init__()
        self.maxdisp = maxdisp
        self.name = name
        
        # Extracting Features
        self.feature_extraction = FeatureExtraction(in_features=3, out_features=32, groups=8)
                 
        #conv3d
        self.conv3d_1=nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3d_1=nn.BatchNorm3d(32)
        self.conv3d_2=nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3d_2=nn.BatchNorm3d(32)
        
        self.conv3d_3=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3d_3=nn.BatchNorm3d(64)
        self.conv3d_4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3d_4=nn.BatchNorm3d(64)
        self.conv3d_5=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3d_5=nn.BatchNorm3d(64)
                 
        #ThreeD Conv Blocks
        self.block_3d_1 = ThreeDConv(in_channels=64, channels=64, stride=2)         
        self.block_3d_2 = ThreeDConv(in_channels=64, channels=64, stride=2)
        self.block_3d_3 = ThreeDConv(in_channels=64, channels=64, stride=2)
        self.block_3d_4 = ThreeDConv(in_channels=64, channels=128, stride=2)   
                 
        #deconvolution 3D
        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.debn1=nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # salih: burada output_padding fantastiko oldu
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.debn4 = nn.BatchNorm3d(32)
                 
        #last deconv3d
        self.deconv5 = nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1) # salih: burada stride 1 idi 2 yaptim
          
    def forward(self, left_img, right_img):   
        
        out_l = self.feature_extraction(left_img)
        out_r = self.feature_extraction(right_img)
        
        cost_volum = self.cost_volume(out_l, out_r)
        
        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volum)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))
        
        #Conv 3D Blocks
        conv3d_block_1 = self.block_3d_1(cost_volum)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volum)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)
        
        #deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4))+conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d))+conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d))+conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d))+conv3d_out)

        #last deconv3d
        deconv3d = self.deconv5(deconv3d)
        #out = torch.squeeze(deconv3d, dim=1)
        out = F.softmax(deconv3d, dim=2)
        
        # weight&sum part  
        idx = torch.arange(0, out.shape[2], dtype=torch.float).to(out.device)
        out = torch.tensordot(out, idx, dims=([2,],[0,]))
        return out
                            
    """
    Concatenating features across channel dimension for each disparity level 
    """
    def cost_volume(self, left_feats, right_feats):

        batch = left_feats.shape[0]
        C = left_feats.shape[1]
        H = left_feats.shape[2]
        W = left_feats.shape[3]

        cost = torch.Tensor(batch, C*2, self.maxdisp//4, H, W).to(left_feats.device)

        for i in range(self.maxdisp//4):
            if(i==0):
                cost[:, :C, i, :, :] = left_feats
                cost[:, C:, i, :, :] = right_feats
            else:
                cost[:, :C, i, :, i:] = left_feats[:, :, :, i:]
                cost[:, C:, i, :, i:] = right_feats[:, :, :, :-i]
        return cost