from __future__ import annotations
import difflib
from inspect import Parameter
import os.path as osp
import xml.etree.ElementTree as ElementTree
import numpy as np

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
    
    def __init__(self, classes):
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