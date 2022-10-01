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