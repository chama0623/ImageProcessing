import torch

def point_form(boxes):
    """DBoxの情報[cx, cy, width, height]を[xmin, ymin, xmax, ymax]に変換する

    Args:
        boxes(Tensor): DBoxの情報(DBoxの数, [cx, cy, width, height])
    Returns:
        boxes(Tensor): DBoxの情報(DBoxの数, [xmin, ymin, xmax, ymax])
    """
    return torch.cat(
        (boxes[:, :2]-boxes[:, 2:]/2,
        boxes[:, :2]+boxes[:, 2:]/2),
        1
    )
    
def intersect(box_a, box_b):
    """2つのボックスが重なる面積を計算する

    Args:
        box_a (Tensor): (DBoxの数, [xmin, ymin, xmax, ymax])
        box_b (Tensor): (DBoxの数, [xmin, ymin, xmax, ymax])
    """
    
    A = box_a.size(0)
    B = box_b.size(0)
    
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A,B,2),
        box_b[:, 2:].unsqueeze(0).expand(A,B,2)
    )
    
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A,B,2),
        box_b[:, :2].unsqueeze(0).expand(A,B,2)    
    )
    
    inter = torch.clamp((max_xy-min_xy), min=0)
    
    return inter[:,:,0]*inter[:,:,1]

def jaccard(box_a, box_b):
    """IoUを計算する関数

    Args:
        box_a (Tensor): (DBoxの数, [xmin, ymin, xmax, ymax])
        box_b (Tensor): (DBoxの数, [xmin, ymin, xmax, ymax])
    Returns:
        (Tensor): BBoxとすべてのDBoxの組み合わせにおけるIoU(box_aのボックス数, box_bのボックス数)
    """
    inter = intersect(box_a, box_b)
    
    area_a = ((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(1).expand_as(inter)
    
    union = area_a + area_b - inter
    
    return inter/union

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """教師データloc, confを作成する

    Args:
        threshold (float): IoUの閾値
        truths (Tensor): ミニバッチの現在の画像におけるDBoxの座標情報(DBoxの数, [xmin, ymin, xmax, ymax])
        priors (Tensor): DBoxの情報(8732, [xmin, ymin, xmax, ymax])
        variances (list): DBoxを変形するオフセット値を計算するときに使用する係数[0.1, 0.2]
        labels (list[int]): 正解ラベルのリスト[BBox1のラベル, BBox1のラベル, ...]
        loc_t (Tensor): 各DBoxに一番近い正解のBBoxのラベルを格納するための3階テンソル(batch_size, 8732,4)
        conf_t (Tensor): 各DBoxに一番近い正解のBBoxのラベルを格納するための3階テンソル(batch_size, 8732,classes_num)
        idx (int): 現在のミニバッチのインデックス
    """
    
    overlaps = jacard(truths, point_form(priors))
    
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
        
    matches = truths[best_prior_idx]
    
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    
    conf_t[idx] = conf