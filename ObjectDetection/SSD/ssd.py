import torch.nn as nn

def make_vgg():
    """generate vgg model
    
    Returns:
        (nn.ModuleList): vgg module list
    """
    layers = [] # list of module
    in_channels = 3 # RGB
    
    # チャネル数のリスト
    # M, MCはPooling
    cfg = [64, 64, "M", # vgg1
           128, 128, "M", # vgg2
           256, 256, 256, "MC", # vgg3
           512, 512, 512, "M", # vgg4
           512, 512, 512, # vgg5
           ]
    
    for v in cfg:
        if v=="M": # Pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v=="MC": # Pooling(奇数width, heightの対応)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else: # Conv
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    # Pooling(vgg5)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    # vgg6
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(512, 1024, kernel_size=1)
    
    layers += [pool5, 
               conv6, nn.ReLU(inplace=True), 
               conv7, nn.ReLU(inplace=True)]
    
    return nn.ModuleList(layers)

def make_extras():
    """generate extras model
    
    Returns:
    (nn.ModuleList) : extras module list
    """
    layers = []
    in_channels = 1024 # vggから出力される画像データのチャネル数
    
    cfg = [256, 512, # extras1
           128, 256, # extras2
           128, 256, # extras3
           128, 256] # extras4
    
    # extras1
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    
    # extras2
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    
    # extras3
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    
    # extras4
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    return nn.ModuleList(layers)