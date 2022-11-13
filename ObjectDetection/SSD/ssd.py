from re import I
from statistics import variance
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F
from itertools import product as product
from math import sqrt as sqrt
from match import match

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

def nonmaximum_suppress(boxes, scores, overlap=0.5, top_k=200):
    
    count=0
    keep = scores.new(scores.size(0)).zero_().long()
    
    # BBoxの面積計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = torch.mul(x2-x1, y2-y1) 
    
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    # scoresを昇順に並べ替える
    v, idx = scores.sort(0)
    # 上位200個を抽出
    idx = idx[-top_k:]
    
    # idxの要素数が0でないとき
    while idx.numel > 0:
        i = idx[-1] # 確信度が最大のインデックス
        keep[count]=i
        count+=1
        
        # 終了判定:idxの要素数が1のとき
        if idx.size(0) == 1:
            break
        
        # 末尾の要素を除外
        idx = idx[:-1]

        # idxに対応させたxmin, ymin, xmax, ymaxを抽出
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)
        
        # idxのBBoxのxmin, ymin, xmax, ymaxを確信度が最大のBBoxの値まで切り詰める
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])
        
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        # 2つの面積の集合積を計算
        inter = tmp_w*tmp_h
        
        # areaからidxに残っている全てのBBoxを取得
        rem_areas = torch.index_select(area, 0, idx)
        # 2つの面積の集合和を計算
        union = (rem_areas - inter) + area[i]
        # IoUを計算
        IoU = inter/union
        
        # overlap以下のIoUのidxだけ残す
        idx = idx[IoU.le(overlap)]
        
    return keep, count

class Detect(Function):
    @staticmethod
    def forward(ctx, loc_data, conf_data, dbox_list):
        ctx.softmax = nn.Softmax(dim=-1)
        # BBoxを抽出するときの閾値
        ctx.conf_thresh = 0.01
        # NMSを実施するBBox数の上限
        ctx.top_k = 200
        # IoUの閾値
        ctx.nms_thresh = 0.45
        
        # ミニバッチサイズ
        batch_num = loc_data.size(0)
        # クラス数
        classes_num = conf_data.size(2)
        
        # ソフトマックス関数をクラスに実行
        conf_data = ctx.softmax(conf_data)
        conf_preds = conf_data.transpose(2,1)
        
        # 出力Tensorを用意
        output = torch.zeros(batch_num, classes_num, ctx.top_k, 5)
        
        for i in range(batch_num):
            decoded_boxes = decode(loc_data[i], dbox_list)
            
            conf_scores = conf_preds[i].clone()
            
            for cl in range(1, classes_num):
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)
                
                scores = conf_scores[cl][c_mask]
                
                if scores.nelement() == 0:
                    continue
                
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                
                boxes = decoded_boxes[l_mask].view(-1, 4)
                
                ids, count = nonmaximum_suppress(boxes, scores, ctx.nms_thresh, ctx.top_k)
                
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)
                
        return output
    
class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        
        self.phase = phase
        self.class_num = cfg["classes_num"]
        
        # SSDネットワークを生成
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc = make_loc(cfg["dbox_num"])
        self.conf = make_conf(cfg["classes_num"], cfg["dbox_num"])
        
        # DBoxの[cx, cy, width, height]を格納したTensor(8732, 4)を取得
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        if phase == "test": # 推論モードのとき
            self.detect = Detect.apply
            
        def forward(self, x):
            """forward propagation

            Args:
                x (Tensor): imageの4階テンソル(batch_size, 3, 300, 300)
            """
            
            out_list = list() # out1～out6を格納
            loc = list() # locの出力を格納
            conf = list() # confの出力を格納
            
            # out1
            for k in range(23):
                x = self.vgg[k](x)
                
            out1 = self.L2Norm(x)
            out_list.append(out1)
            
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
                
            out_list.append(x)
            
            # out3～out6
            for k,v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 ==1:
                    out_list.append(x)
                    
            # loc, conf
            for (x, l, c) in zip(out_list, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
                
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], l)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], l)
            
            loc = loc.view(loc.size(0), -1, 4)
            conf = conf.view(conf.size(0), -1, self.classes_num)
            
            output = (loc, conf, self.dbox_list)
            
            if self.phase == "test":
                return self.detect(output[0], output[1], output[2])
            else:
                return output
            
class MultiBoxLoss(nn.Module):
    """SSDの損失関数クラス

    """
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device
        
    def forward(self, predictions, targets):
        """損失関数を適用してlocとconfの出力の誤差を求める

        Args:
            predictions (tuple): SSDの出力(loc, conf, DBox)
            targets (Tensor): (batch_size, 物体数, [xmin, ymin, xmax, ymax, label_index])
        """
        loc_data, conf_data, dbox_list = predictions
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
        
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)
        
        for idx in range(num_batch):
            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)
            dbox = dbox_list.to(self.device)
            variance = [0.1, 0.2]
            
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)
            
            pos_mask = conf_t_label > 0
            pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
            
            loc_p = loc_data[pos_idx].view(-1, 4)
            
            loc_t = loc_t[pos_idx].view(-1, 4)
            
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")
            
            batch_conf = conf_data.view(-1, num_classes)
            
            loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")
            
            num_pos = pos_mask.long().sum(1, keepdim=True)
            
            loss_c = loss_c.view(num_batch, -1)
            
            loss_c[pos_mask] = 0
            
            _, loss_idx = loss_c.sort(1, descending=True)
            
            _, idx_rank = loss_idx.sort(1)
            
            num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
            
            neg_mask = idx_rank < (num_neg).expand_as(idx_rank)
            
            pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
            
            neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
            
            conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
            
            conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]
            
            loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction="sum")
            
            N = num_pos.sum()
            
            loss_l /= N
            
            loss_c /= N
            
            return loss_l, loss_c