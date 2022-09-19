from inspect import Parameter
import os.path as osp

def make_filepath_list(rootpath):
    """make list of images and annotations
    
    Args:
    rootpath(str) : データフォルダのルートパス
    
    Returns:
    train_img_list : 訓練用イメージのパスリスト
    train_anno_list : 訓練用アノテーションのパスリスト
    val_img_list : 検証用イメージのパスリスト
    val_anno_list : 検証用アノテーションのパスリスト
    """
    
    imgpath_templete = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_templete = osp.join(rootpath, "Annotations", "%s.xml")
    
    train_id_names = osp.join(rootpath+"ImageSets/Main/train.txt")
    val_id_names = osp.join(rootpath, "ImageSets/Main/val.txt")
    
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip() # 空白スペースと改行削除
        img_path = (imgpath_templete % file_id)
        anno_path = (annopath_templete % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_templete % file_id)
        anno_path = (annopath_templete % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list      