import torch
import torch.nn as nn
import torch.nn.init as init
from itertools import product as product
from math import sqrt as sqrt

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

def make_loc(dbox_num=[4, 6, 6, 6, 4, 4]):
    """generate loc network

    Args:
        dbox_num(list) : loc1～loc6までに用意されているoffsetの数
    
    Returns:
        (nn.MoudleList) : locモジュールのリスト
    """
    
    loc_layers = []
    
    loc_layers += [nn.Conv2d(512, dbox_num[0]*4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(1024, dbox_num[1]*4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(512, dbox_num[2]*4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[3]*4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[4]*4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[5]*4, kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers)

def make_conf(classes_num=21, dbox_num=[4, 6, 6, 6, 4, 4]):
    """generate conf network

    Args:
        classes_num (int, optional): class number. Defaults to 21.
        dbox (list, optional): loc1～loc6までに用意されているoffsetの数. Defaults to [4, 6, 6, 6, 4, 4].
    
    Returns:
        (nn.MoudleList) : locモジュールのリスト
    """
    
    conf_layers = []
    
    conf_layers += [nn.Conv2d(512, dbox_num[0]*classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, dbox_num[1]*classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, dbox_num[2]*classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[3]*classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[4]*classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[5]*classes_num, kernel_size=3, padding=1)]
    
    return nn.ModuleList(conf_layers)

class L2Norm(nn.Module):
    
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        
        self.weight = nn.parameter(torch.Tensor(input_channels))
        self.scale = scale # 重みの初期値
        self.reset_parameters() # 重みを初期化
        self.eps = 1e-10 # L2ノルムの値値に加算する極小値
        
        def reset_parameters(self):
            """重みをscaleで初期化する
            """
            init.constant_(self.weight, self.scale)
        
    def forward(self, x):
        """L2Normの順伝搬を行う

        Args:
            x (Tensor): (batch_size, 512, 38, 38)

        Returns:
            Tensor: L2ノルムで正規化した後にsacleの重みを適用したテンソル(batch_size, 512, 38, 38)
        """
        
        # normを計算
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        # 正規化
        x = torch.div(x, norm)
        
        # weightを4階テンソルに変形
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # xに重みを適用
        out = weights*x
        return out
    
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.image_size = cfg["input_size"] # 300
        self.feature_maps = cfg["feature_maps"] # [38, 19, 10, 5, 3, 1]
        self.num_priors = len(cfg["feature_maps"]) # 6
        self.steps = cfg["steps"] # DBoxのサイズ [8, 16, 32, 64, 100, 300]
        self.min_sizes = cfg["min_sizes"] # 小さい正方形のサイズ [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg["max_sizes"] # 大きい正方形のサイズ [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = cfg["aspect_ratios"] # アスペクト比[[2], [2,3], [2,3], [2,3], [2], [2]]
        
    def make_dbox_list(self):
        """Default Boxを生成

        Returns:
            Tensor: [cx, cy, width, height]を格納した2階テンソル(8732, 4)
        """
        # [[cx, xy, widh, weight], ...]
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # 300 / [8, 16, 32, 64, 100, 300]
                f_k= self.image_size/ self.steps[k]
                # DBoxの中心座標(正規化)を計算
                cx = (j+0.5)/f_k
                cy = (i+0.5)/f_k
                
                # 小さい正方形
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                
                # 大きい正方形
                s_k_prime = sqrt(s_k*(self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # 長方形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    
        # 2階テンソルに変換
        output = torch.Tensor(mean).view(-1, 4)
        # DBoxの値チェック
        output.clamp_(max=1, min=0)
        return output
    
def decode(loc, dbox_list):
    """デフォルトボックスからバウンディングボックスに変換する関数

    Args:
        loc (Tensor): output of loc(8721, 4)
        dbox_list (Tensor): [[cx, cy, width, height]]のTensor(8732, 4)

    Returns:
        _type_: _description_
    """
    
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:,:2]*0.1*dbox_list[:, 2:],
        dbox_list[:, 2:]*torch.exp(loc[:, 2:]*0.2)
    ), dim=1)
    
    boxes[:, :2] -= boxes[:, 2:]/2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes