import torch.nn as nn
import torch

# various architectures for UNet


def double_conv(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

def conv(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
    )
####################                   DANIEL's DEFAULT UNET                 ######################

class UNet_tail(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()

        self.dconv_d1 = double_conv(1, 64)
        self.dconv_d2 = double_conv(64, 128)
        self.dconv_d3 = double_conv(128, 256)
        self.dconv_d4 = double_conv(256, 512)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_u3 = double_conv(512 + 256, 256) # skip connection, so adding 256
        self.dconv_u2 = double_conv(256 + 128, 128)
        self.dconv_u1 = double_conv(128 + 64, 64)

        self.conv1 = conv(64, 64)
        self.conv2 = conv(64, 64)
        self.conv3 = conv(64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_d1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_d2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_d3(x)
        x = self.maxpool(conv3)

        x = self.dconv_d4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim = 1)

        x = self.dconv_u3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_u2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_u1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        out = self.conv_last(x)

        return out

####################                   MY DEFAULT UNET                 ######################

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.dconv_d1 = double_conv(3, 64)
        self.dconv_d2 = double_conv(64, 128)
        self.dconv_d3 = double_conv(128, 256)
        self.dconv_d4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_u3 = double_conv(512 + 256, 256)  # skip connection, so adding 256
        self.dconv_u2 = double_conv(256 + 128, 128)
        self.dconv_u1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1, dilation=1)

    def forward(self, x):
        conv1 = self.dconv_d1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_d2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_d3(x)
        x = self.maxpool(conv3)

        x = self.dconv_d4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_u3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_u2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_u1(x)

        out = self.conv_last(x)

        return out

    
####################                   RESUNET                 ######################

def convrelu(in_chnls, out_chnls, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_chnls, out_channels = out_chnls, kernel_size = kernel, padding = padding),
        nn.BatchNorm2d(num_features = out_chnls),
        nn.ReLU(inplace = True)
    )

"""
ResUNet-> U-Net with resnet encoder.
Inputs:
	layers: layers of the pretrained resnet passed as a list
	classes: number of foreground classes
Example:

	resnet = models.resnet34(pretrained=True)
	layers = list(resnet.children())
	model = ResUNet(layers, 1)
"""

class ResUNet(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()
        
        self.base_layers = layers
        self.layer_0 = nn.Sequential(*self.base_layers[:3])  # ip   3, op 64 features
        self.layer_1 = nn.Sequential(*self.base_layers[3:5]) # ip  64, op 64 features
        self.layer_2 = nn.Sequential(*self.base_layers[5])   # ip  64, op 128 features
        self.layer_3 = nn.Sequential(*self.base_layers[6])   # ip 128, op 256 features
        self.layer_4 = nn.Sequential(*self.base_layers[7])   # ip 256, op 512 features
          
        self.layer_4_1x1 = convrelu(512, 512, 1, 0)
        self.layer_3_1x1 = convrelu(256, 256, 1, 0)
        self.layer_2_1x1 = convrelu(128, 128, 1, 0)
        self.layer_1_1x1 = convrelu(64, 64, 1, 0)                
        self.layer_0_1x1 = convrelu(64, 64, 1, 0)
        
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        
        self.up_conv3 = convrelu(512 + 256, 512, 3, 1)
        self.up_conv2 = convrelu(128 + 512, 256, 3, 1)
        self.up_conv1 = convrelu(64 + 256, 256, 3, 1)
        self.up_conv0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_same_size = convrelu(3, 64, 3, 1)
        self.conv_same_size2 = convrelu(64, 64, 3, 1)
        self.conv_same_size3 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, input):
        
        input_ss = self.conv_same_size(input)
        input_ss = self.conv_same_size2(input_ss)
        
        layer0 = self.layer_0(input) 
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)
        
        layer4 = self.layer_4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer_3_1x1(layer3)
        x = torch.cat([x, layer3], dim = 1)
        x = self.up_conv3(x)
        
        x = self.upsample(x)
        layer2 = self.layer_2_1x1(layer2)
        x = torch.cat([x, layer2], dim = 1)
        x = self.up_conv2(x)
        
        
        x = self.upsample(x)
        layer1 = self.layer_1_1x1(layer1)
        x = torch.cat([x, layer1], dim = 1)
        x = self.up_conv1(x)
        
        x = self.upsample(x)
        layer0 = self.layer_0_1x1(layer0)
        x = torch.cat([x, layer0], dim = 1)
        x = self.up_conv0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, input_ss], dim = 1)
        x = self.conv_same_size3(x)
        
        out = self.conv_last(x)
        
        return out
    
############################     ATTENTION RESUNET    #####################################    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttnResUNet(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()
        
        self.base_layers = layers
        self.layer_0 = nn.Sequential(*self.base_layers[:3]) # ip   3, op 64 features
        self.layer_0_1x1 = convrelu(64, 64, 1, 0)
        
        self.layer_1 = nn.Sequential(*self.base_layers[3:5]) # ip  64, op 64 features
        self.layer_1_1x1 = convrelu(64, 64, 1, 0)   
        
        self.layer_2 = nn.Sequential(*self.base_layers[5])   # ip  64, op 128 features
        self.layer_2_1x1 = convrelu(128, 128, 1, 0)
        
        self.layer_3 = nn.Sequential(*self.base_layers[6])   # ip 128, op 256 features
        self.layer_3_1x1 = convrelu(256, 256, 1, 0)
        
        self.layer_4 = nn.Sequential(*self.base_layers[7])   # ip 256, op 512 features
        self.layer_4_1x1 = convrelu(512, 512, 1, 0)
        
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        
        self.attn3 = Attention_block(256, 512, 512)
        self.up_conv3 = convrelu(512 + 512, 512, 3, 1)
        
        self.attn2 = Attention_block(128, 512, 512)
        self.up_conv2 = convrelu(1024, 256, 3, 1)
        
        self.attn1 = Attention_block(64, 256, 256)
        self.up_conv1 = convrelu(512, 256, 3, 1)
        
        self.attn0 = Attention_block(64, 256, 256)
        self.up_conv0 = convrelu(512, 128, 3, 1)
        
        self.conv_same_size = convrelu(3, 64, 3, 1)
        self.conv_same_size2 = convrelu(64, 64, 3, 1)
        self.conv_same_size3 = convrelu(128 + 64, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, input):
        
        input_ss = self.conv_same_size(input)
        input_ss = self.conv_same_size2(input_ss)
        
        layer0 = self.layer_0(input) 
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)
        
        layer4 = self.layer_4_1x1(layer4)   # op 512 channels
        
        x = self.upsample(layer4)           # op 512 channels
        layer3 = self.layer_3_1x1(layer3)   # op 256 channels
        x4 = self.attn3(layer3, x)          # ip(256, 512, 256)
        x = torch.cat([x, x4], dim = 1) # SKIP
        x = self.up_conv3(x)
        
        x = self.upsample(x)
        layer2 = self.layer_2_1x1(layer2)
        x3 = self.attn2(layer2, x)
        x = torch.cat([x, x3], dim = 1) # SKIP
        x = self.up_conv2(x)
        
        
        x = self.upsample(x)
        layer1 = self.layer_1_1x1(layer1)
        x2 = self.attn1(layer1, x)
        x = torch.cat([x, x2], dim = 1) # SKIP
        x = self.up_conv1(x)
        
        x = self.upsample(x)
        layer0 = self.layer_0_1x1(layer0)
        x1 = self.attn0(layer0, x)
        x = torch.cat([x, x1], dim = 1) # SKIP
        x = self.up_conv0(x)
        
        x = self.upsample(x)
        
        x = torch.cat([x, input_ss], dim = 1) # SKIP
        x = self.conv_same_size3(x)
        
        out = self.conv_last(x)
        
        return out    
    
    
class AttnResUNet2(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()
        
        self.base_layers = layers
        self.layer_0 = nn.Sequential(*self.base_layers[:3]) # ip   3, op 64 features
        self.layer_0_1x1 = convrelu(64, 64, 1, 0)
        
        self.layer_1 = nn.Sequential(*self.base_layers[3:5]) # ip  64, op 64 features
        self.layer_1_1x1 = convrelu(64, 64, 1, 0)   
        
        self.layer_2 = nn.Sequential(*self.base_layers[5])   # ip  64, op 128 features
        self.layer_2_1x1 = convrelu(128, 128, 1, 0)
        
        self.layer_3 = nn.Sequential(*self.base_layers[6])   # ip 128, op 256 features
        self.layer_3_1x1 = convrelu(256, 256, 1, 0)
        
        self.layer_4 = nn.Sequential(*self.base_layers[7])   # ip 256, op 512 features
        self.layer_4_1x1 = convrelu(512, 512, 1, 0)
        
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        
        self.attn3 = Attention_block(256, 512, 512)
        self.up_conv3 = convrelu(512 + 512, 256, 3, 1)
        
        self.attn2 = Attention_block(128, 256, 256)
        self.up_conv2 = convrelu(512, 128, 3, 1)
        
        self.attn1 = Attention_block(64, 128, 128)
        self.up_conv1 = convrelu(256, 64, 3, 1)
        
        self.attn0 = Attention_block(64, 64, 64)
        self.up_conv0 = convrelu(128, 64, 3, 1)
        
        self.conv_same_size = convrelu(3, 64, 3, 1)
        self.conv_same_size2 = convrelu(64, 64, 3, 1)
        self.conv_same_size3 = convrelu(128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, input):
        
        input_ss = self.conv_same_size(input)
        input_ss = self.conv_same_size2(input_ss)
        
        layer0 = self.layer_0(input) 
        layer1 = self.layer_1(layer0)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)
        
        layer4 = self.layer_4_1x1(layer4)   # op 512 channels
        
        x = self.upsample(layer4)           # op 512 channels
        layer3 = self.layer_3_1x1(layer3)   # op 256 channels
        x4 = self.attn3(layer3, x)          # ip(256, 512, 256)
        x = torch.cat([x, x4], dim = 1) # SKIP
        x = self.up_conv3(x)
        
        x = self.upsample(x)
        layer2 = self.layer_2_1x1(layer2)
        x3 = self.attn2(layer2, x)
        x = torch.cat([x, x3], dim = 1) # SKIP
        x = self.up_conv2(x)
        
        
        x = self.upsample(x)
        layer1 = self.layer_1_1x1(layer1)
        x2 = self.attn1(layer1, x)
        x = torch.cat([x, x2], dim = 1) # SKIP
        x = self.up_conv1(x)
        
        x = self.upsample(x)
        layer0 = self.layer_0_1x1(layer0)
        x1 = self.attn0(layer0, x)
        x = torch.cat([x, x1], dim = 1) # SKIP
        x = self.up_conv0(x)
        
        x = self.upsample(x)
        
        x = torch.cat([x, input_ss], dim = 1) # SKIP
        x = self.conv_same_size3(x)
        
        out = self.conv_last(x)
        
        return out
    
####################                   INCEPTION UNET                 ######################

class Inception1(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception1, self).__init__()
        self.branch1x1 = conv(in_channels, pool_features, kernel_size=1) # 1

        self.branch5x5_1 = conv(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv(48, pool_features, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv(96, pool_features, kernel_size=3, padding=1)

        self.branch_pool = conv(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

    
class Inception2(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception2, self).__init__()
        self.branch1x1 = conv(in_channels, pool_features, kernel_size=1) # 1

        self.branch5x5_1 = conv(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv(48, pool_features, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv(96, pool_features, kernel_size=3, padding=1)

        self.branch_pool = conv(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Inception3(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception3, self).__init__()
        self.branch1x1 = conv(in_channels, pool_features, kernel_size=1) # 1

        self.branch5x5_1 = conv(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv(48, pool_features, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv(96, pool_features, kernel_size=3, padding=1)

        self.branch_pool = conv(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Inception4(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception4, self).__init__()
        self.branch1x1 = conv(in_channels, pool_features, kernel_size=1) # 1

        self.branch5x5_1 = conv(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv(48, pool_features, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv(96, pool_features, kernel_size=3, padding=1)

        self.branch_pool = conv(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class incepUnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.dconv_d1 = Inception1(1, 16)
        self.dconv_d2 = Inception2(64, 32)
        self.dconv_d3 = Inception3(128, 64)
        self.dconv_d4 = Inception4(256, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_u3 = double_conv(512 + 256, 256)  # skip connection, so adding 256
        self.dconv_u2 = double_conv(256 + 128, 128)
        self.dconv_u1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1, dilation=1)

    def forward(self, x):
        conv1 = self.dconv_d1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_d2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_d3(x)
        x = self.maxpool(conv3)

        x = self.dconv_d4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_u3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_u2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_u1(x)

        out = self.conv_last(x)

        return out