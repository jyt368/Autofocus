
import math
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision
import functools # used by RRDBNet


def GetModel(opt):
    
    if opt.model.lower() == 'rcan':
        net = RCAN(opt)
    elif opt.model.lower() == 'rcan_zernike':
        net = RCAN_zernike(opt) 
    elif opt.model.lower() == 'yolo':
        net = YoloModel(opt)     

    else:
        print("model undefined")    
        return None
    
    if not opt.cpu:
        net.cuda()
        if opt.multigpu:
            net = nn.DataParallel(net)

    return net

def normalizationTransforms(normtype):
    if normtype.lower() == 'div2k':
        normalize = MeanShift(1, [0.4485, 0.4375, 0.4045], [0.2436, 0.2330, 0.2424])
        unnormalize = MeanShift(1, [-1.8411, -1.8777, -1.6687], [4.1051, 4.2918, 4.1254])
        print('using div2k normalization')
    elif normtype.lower() == 'pcam':
        normalize = MeanShift(1, [0.6975, 0.5348, 0.688], [0.2361, 0.2786, 0.2146])
        unnormalize = MeanShift(1, [-2.9547, -1.9198, -3.20643], [4.2363, 3.58972, 4.66049])
        print('using pcam normalization')
    elif normtype.lower() == 'div2k_std1':
        normalize = MeanShift(1, [0.4485, 0.4375, 0.4045], [1,1,1])
        unnormalize = MeanShift(1, [-0.4485, -0.4375, -0.4045], [1,1,1])
        print('using div2k normalization with std 1')
    elif normtype.lower() == 'pcam_std1':
        normalize = MeanShift(1, [0.6975, 0.5348, 0.688], [1,1,1])
        unnormalize = MeanShift(1, [-0.6975, -0.5348, -0.688], [1,1,1])
        print('using pcam normalization with std 1')
    else:
        print('not using normalization')
        return None, None
    return normalize, unnormalize


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)




# ----------------------------------- RCAN ------------------------------------------

## Channel Attention Reduction (CA) Layer
class CAReductionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAReductionLayer, self).__init__()
        # global average pooling: feature --> point
        self.reduce = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(channel, channel)
        )


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        x = self.reduce(x)
        return x * y

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)

        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, opt): 
        super(RCAN, self).__init__()
        n_resgroups = opt.n_resgroups
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3
        reduction = opt.reduction
        act = nn.ReLU(True)
        self.narch = opt.narch
        
        if not opt.norm == None:
            self.normalize, self.unnormalize = normalizationTransforms(opt.norm)
        else:
            self.normalize, self.unnormalize = None, None


        # define head module
        if self.narch == 0:
            modules_head = [conv(opt.nch_in, n_feats,  5)]
            self.head = nn.Sequential(*modules_head)
        else:
            self.head0 = conv(1, n_feats, kernel_size)
            self.head02 = conv(n_feats, n_feats, kernel_size)
            self.head1 = conv(1, n_feats, kernel_size)
            self.head12 = conv(n_feats, n_feats, kernel_size)
            self.head2 = conv(1, n_feats, kernel_size)
            self.head22 = conv(n_feats, n_feats, kernel_size)
            self.head3 = conv(1, n_feats, kernel_size)
            self.head32 = conv(n_feats, n_feats, kernel_size)
            self.head4 = conv(1, n_feats, kernel_size)
            self.head42 = conv(n_feats, n_feats, kernel_size)
            self.head5 = conv(1, n_feats, kernel_size)
            self.head52 = conv(n_feats, n_feats, kernel_size)
            self.head6 = conv(1, n_feats, kernel_size)
            self.head62 = conv(n_feats, n_feats, kernel_size)
            self.head7 = conv(1, n_feats, kernel_size)
            self.head72 = conv(n_feats, n_feats, kernel_size)
            self.head8 = conv(1, n_feats, kernel_size)
            self.head82 = conv(n_feats, n_feats, kernel_size)
            self.combineHead = conv(9*n_feats, n_feats, kernel_size)

            

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if opt.scale == 1:
            if opt.task == 'segment':
                modules_tail = [nn.Conv2d(n_feats, opt.nch_out, 1)]
            else:
                modules_tail = [conv(n_feats, opt.nch_out, kernel_size)]
        else:
            modules_tail = [
                Upsampler(conv, opt.scale, n_feats, act=False),
                conv(n_feats, opt.nch_out, kernel_size)]
        
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if not self.normalize == None:
            x = self.normalize(x)

        if self.narch == 0:
            x = self.head(x)
        else:
            x0 = self.head02(self.head0(x[:,0:0+1,:,:]))
            x1 = self.head12(self.head1(x[:,1:1+1,:,:]))
            x2 = self.head22(self.head2(x[:,2:2+1,:,:]))
            x3 = self.head32(self.head3(x[:,3:3+1,:,:]))
            x4 = self.head42(self.head4(x[:,4:4+1,:,:]))
            x5 = self.head52(self.head5(x[:,5:5+1,:,:]))
            x6 = self.head62(self.head6(x[:,6:6+1,:,:]))
            x7 = self.head72(self.head7(x[:,7:7+1,:,:]))
            x8 = self.head82(self.head8(x[:,8:8+1,:,:]))
            x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8), 1)
            x = self.combineHead(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if not self.unnormalize == None:
            x = self.unnormalize(x)

        return x 

## Residual Channel Attention Network for zernike (RCAN_zernike)
class RCAN_zernike(nn.Module):
    def __init__(self, opt): 
        super(RCAN_zernike, self).__init__()
        n_resgroups = opt.n_resgroups
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3
        reduction = opt.reduction
        act = nn.ReLU(True)
        self.narch = opt.narch
        
        if not opt.norm == None:
            self.normalize, self.unnormalize = normalizationTransforms(opt.norm)
        else:
            self.normalize, self.unnormalize = None, None

        # define head module
       
        modules_head = [conv(opt.nch_in, n_feats,  5)]
        self.head = nn.Sequential(*modules_head)         

        # define body module
        modules_body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks)]
        
        for g in range(n_resgroups):

            modules_body.append(CAReductionLayer(n_feats, reduction))
            modules_body.append(nn.BatchNorm2d(n_feats))
            modules_body.append(ResidualGroup(conv, n_feats, kernel_size, reduction*(g+1), act=act, res_scale=1, n_resblocks=n_resblocks))
            
        

        # define tail module
        if opt.scale == 0:

            # use down module to reduce output to 1 number
            modules_tail = [nn.AdaptiveAvgPool2d(1)]      
            modules_tail.append(nn.Flatten())
            modules_tail.append(nn.Linear(n_feats,  opt.n_zernike))     

        else:   
            
            modules_tail.append(nn.AdaptiveAvgPool2d(1))
            modules_tail.append(nn.Flatten())
            modules_tail.append(nn.Linear((g+2)*n_feats, opt.n_zernike))


  
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if not self.normalize == None:
            x = self.normalize(x)

        if self.narch == 0:
            x = self.head(x)
        else:
            x0 = self.head02(self.head0(x[:,0:0+1,:,:]))
            x1 = self.head12(self.head1(x[:,1:1+1,:,:]))
            x2 = self.head22(self.head2(x[:,2:2+1,:,:]))
            x3 = self.head32(self.head3(x[:,3:3+1,:,:]))
            x4 = self.head42(self.head4(x[:,4:4+1,:,:]))
            x5 = self.head52(self.head5(x[:,5:5+1,:,:]))
            x6 = self.head62(self.head6(x[:,6:6+1,:,:]))
            x7 = self.head72(self.head7(x[:,7:7+1,:,:]))
            x8 = self.head82(self.head8(x[:,8:8+1,:,:]))
            x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8), 1)
            x = self.combineHead(x)

        res = self.body(x)
        

        x = self.tail(res)

        if not self.unnormalize == None:
            x = self.unnormalize(x)

        return x 


## Yolo Network for zernike (RCAN_zernike)
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cbl = nn.Sequential(
            conv,
            bn,
            
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.cbl(x)
    
class Bottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x

class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
    """
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)   

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))




class YoloModel(nn.Module):
    def __init__(self, opt): 
        super(YoloModel, self).__init__()
        first_out = opt.n_feats
        n_in_channels = opt.nch_in
        out_channels = opt.n_zernike


        # define backbone module
        
        modules_backbone = [CBL(n_in_channels, first_out, kernel_size=6, stride=1, padding=2)] # [256 256 128]
        modules_backbone.append(C3(in_channels=first_out, out_channels=first_out, width_multiple=0.5, depth=2)) # [256 256 128]

        modules_backbone.append(CBL(in_channels=first_out, out_channels=first_out*2, kernel_size=3, stride=2, padding=1)) # [128 128 256]
        modules_backbone.append(C3(in_channels=first_out*2, out_channels=first_out*2, width_multiple=0.5, depth=4)) # [128 128 256]
        # remove x here [128 128 256]
        modules_backbone.append(CBL(in_channels=first_out*2, out_channels=first_out*4, kernel_size=3, stride=2, padding=1)) # [64 64 512]
        modules_backbone.append(C3(in_channels=first_out*4, out_channels=first_out*4, width_multiple=0.5, depth=6)) # [64 64 512]
        # remove x here [64 64 512]
        modules_backbone.append(CBL(in_channels=first_out*4, out_channels=first_out*8, kernel_size=3, stride=2, padding=1)) # [32 32 1024]
        modules_backbone.append(C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, depth=2)) # [32 32 1024]
        modules_backbone.append(SPPF(in_channels=first_out*8, out_channels=first_out*8)) # [32 32 1024]

        self.backbone = nn.Sequential(*modules_backbone)  

        # define neck module
        modules_neck = [CBL(in_channels=first_out*8, out_channels=first_out*4, kernel_size=1, stride=1, padding=0)] # [32 32 512]
        # remove here [32 32 512]
        # upsample here [64 64 512]
        # concat here [64 64 512+512]
        modules_neck.append(C3(in_channels=first_out*8, out_channels=first_out*4, width_multiple=0.25, depth=2, backbone=False)) # [64 64 512]
        modules_neck.append(CBL(in_channels=first_out*4, out_channels=first_out*2, kernel_size=1, stride=1, padding=0)) # [32 32 256]
        # remove here [64 64 256]
        # upsample here [128 128 256]
        # concat here [128 128 256+256]
                
        modules_neck.append(C3(in_channels=first_out*4, out_channels=first_out*2, width_multiple=0.25, depth=2, backbone=False)) # [128 128 256]
        modules_neck.append(CBL(in_channels=first_out*2, out_channels=first_out*2, kernel_size=3, stride=2, padding=1)) # [64 64 256]

        # concat here [64 64 256+256]
        modules_neck.append(C3(in_channels=first_out*4, out_channels=first_out*4, width_multiple=0.5, depth=2, backbone=False)) # [64 64 512]
        modules_neck.append(CBL(in_channels=first_out*4, out_channels=first_out*4, kernel_size=3, stride=2, padding=1)) # [32 32 512]
        # concat here [32 32 512+512]

        modules_neck.append(C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, depth=2, backbone=False)) # [32 32 1024]
       

        self.neck = nn.Sequential(*modules_neck)

        # define tail module
        modules_tail = [nn.AdaptiveAvgPool2d(1)]      
        modules_tail.append(nn.Flatten())
        modules_tail.append(nn.Linear(first_out*8, out_channels))     

        self.tail = nn.Sequential(*modules_tail)
        self.upsize = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        backbone_connection = []
        neck_connection = []

        
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in [3, 5]:
                backbone_connection.append(x)


        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = self.upsize(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            else:
                x = layer(x)

        x = self.tail(x)

        return x 















# ------------------ Alternative UNet implementation (batchnorm. outcommented)
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_ch,in_ch, 2, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

