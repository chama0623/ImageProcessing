from __future__ import annotations
import difflib
from inspect import Parameter
import os.path as osp
import xml.etree.ElementTree as ElementTree
import numpy as np
from augmentations import Compose, ConvertColor, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
import torch
import torch.utils.data as data
import cv2

def make_filepath_list(rootpath):
    """make list of images and annotations

    Args:
    rootpath(str): データフォルダのルートパス
    
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

class GetBBoxAndLabel:
    """annotation one image
    """
    
    def __init__(self, classes): #__method__ をダンダーと呼ぶ
        """constructor

        Args:
            classes (list): list of classes
        """

        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        """method execute from instance

        Args:
            xml_path (str): path of xml
            width (int): width of image (for normalize)
            height (int): height of image (for normalize)
        
        Returns(ndarray):
        [[xmin, ymin, xmax, ymax, index], ...]
        number of element equals object in the image
        """
        
        annotations = []
        xml = ElementTree.parse(xml_path).getroot() # get path
        
        # get info from xml
        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text) # get difficult
            if difficult == 1:
                continue # not append to annotations list
            
            bndbox = []
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            grid = ["xmin", "ymin", "xmax", "ymax"]
            
            for gr in (grid):
                axis_value = int(bbox.find(gr).text)-1 # origin adjustment
                # normalization
                if gr == "xmin" or gr == "xmax":
                    axis_value /= width
                else:
                    axis_value /= height
                
                bndbox.append(axis_value)
                
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            annotations += [bndbox]
        
        return np.array(annotations)
    
class DataTransform(object):
    """preprocess data

    Args:
        object (_type_): _description_
    """
    
    def __init__(self, input_size, color_mean):
        self.transform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        return self.transform[phase](img, boxes, labels)
    
class PreprocessVOC2012(data.Dataset):
    """extends torch.utils.data.Dataset(abstract)
       doc : https://pytorch.org/docs/stable/data.html
       DataTransformでVOC2012データセットを前処理して次の返す
       
       前処理後のイメージ[R, G, B](Tensor)
       BBoxとlabel(ndarray)
       イメージの高さ・幅(int)

    Args:
        data (_type_): _description_
    """
    
    def __init__(self, img_list, anno_list, phase, transform, get_bbox_label):
        """constructor

        Args:
            img_list (list): list of image file path
            anno_list (list): list of annotation file path
            phase (str): "train" or "test"
            transform (object): class DataTransform
            get_bbox_label (object): class GetBBoxAndLabel
        """
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.get_bbox_label = get_bbox_label
        
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, index):
        """データの回数だけイテレートする

        Args:
            index (int): index of image
                
        Returns:
            im(Tensor): 前処理後のイメージを格納した3階テンソル
            (3, 高さのピクセル, 幅のピクセル)
            bl(ndarray): BBoxとlabelの2次元リスト
        """
            
        im, bl, _, _ = self.pull_item(index)
        return im, bl
        
    def pull_item(self, index):
        """前処理後のテンソル形式のイメージデータ, アノテーション, イメージの高さ, 幅を取得する

        Args:
            index (int): index of image
            
        Returns:
            img(Tensor): 前処理後のイメージ(3, 高さのピクセル数, 幅のピクセル数)
            boxlbl(ndarray): BBoxとlabelの2次元配列
            height(int): イメージの高さ
            width(int): イメージの幅
        """
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        height, width, _ = img.shape
            
        anno_file_path = self.anno_list[index]
        bbox_label = self.get_bbox_label(anno_file_path, width, height)
        
        img, boxes, labels = self.transform(
            img,
            self.phase,
            bbox_label[:, :4],
            bbox_label[:, 4]
        )
        
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        boxlbl = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, boxlbl, height, width
    
def multiobject_collate_fn(batch):
    """イメージとイメージに対するアノテーションをミニバッチの数だけ生成する

    Args:
        batch (tuple): PreprocessVOC2012クラスの__getitem__関数で返される要素数2のタプル
    
    Returns:
        imgs(Tensor): 前処理後のイメージ(RGB)をミニバッチの数だけ格納した4階テンソル(batch_size, 3, 300, 300)
        targets(list): [物体数, 5]の2階テンソル
    """
    
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0]) # タプルの第一要素はイメージ
        targets.append(torch.FloatTensor(sample[1])) # タプルの第二要素はB-Boxとlabel
        
    imgs = torch.stack(imgs, dim=0)
    
    return imgs, targets