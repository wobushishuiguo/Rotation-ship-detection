import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image
import  math

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class XMLDataset(CustomDataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """

    def __init__(self, min_size=None, **kwargs):
        super(XMLDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.bmp'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objects = root.find('HRSC_Objects')
                for obj in objects.findall('HRSC_Object'):
                    name = 'ship'
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        ratios_lu_rb = []
        ratios_lu_rb_ignore = []
        pi = 3.1415926
        img_width = float(root.find('Img_SizeWidth').text)
        img_height = float(root.find('Img_SizeHeight').text)
        objects = root.find('HRSC_Objects')
        for obj in objects.findall('HRSC_Object'):
            name = 'ship'
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            #bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            cx = float(obj.find('mbox_cx').text)
            cy = float(obj.find('mbox_cy').text)
            cw = float(obj.find('mbox_w').text)
            ch = float(obj.find('mbox_h').text)
            ang = float(obj.find('mbox_ang').text)
            if ang >= 0:
                if ang > pi/2:
                    ang = ang - pi/2
                w = cw*math.cos(ang)+ch*math.sin(ang)
                h = ch*math.cos(ang)+cw*math.sin(ang)
                r1 = ch*math.sin(ang)/w
                r2 = ch*math.cos(ang)/h
            else:
                ang = -ang
                if ang > pi/2:
                    ang = ang - pi/2
                w = cw*math.cos(ang)+ch*math.sin(ang)
                h = ch*math.cos(ang)+cw*math.sin(ang)
                r1 = cw*math.cos(ang)/w
                r2 = cw*math.sin(ang)/h
            xmin = cx - w/2
            xmax = cx + w/2
            ymin = cy - h/2
            ymax = cy + h/2
            if xmin < 0:
                xmin = 0
            if xmax > img_width:
                xmax = img_width
            if ymin < 0:
                ymin = 0
            if ymax > img_height:
                ymax = img_height
            bbox = [
                int(xmin),
                int(ymin),
                int(xmax),
                int(ymax)
            ]
            ratio_lu_rb = [r1, r2]


            ignore = False
            if self.min_size:
                assert not self.test_mode
                #w = bbox[2] - bbox[0]
                #h = bbox[3] - bbox[1]

                if w < self.min_size or h < self.min_size:
                    ignore = True

            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
                ratios_lu_rb_ignore.append(ratio_lu_rb)
            else:
                bboxes.append(bbox)
                labels.append(label)
                ratios_lu_rb.append((ratio_lu_rb))
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            ratios_lu_rb = np.zeros((0, 2))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
            ratios_lu_rb = np.array(ratios_lu_rb, ndmin=2)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
            ratios_lu_rb_ignore = np.zeros((0, 2))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
            ratios_lu_rb_ignore = np.array(ratios_lu_rb_ignore, ndmin=2)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            ratios_lu_rb=ratios_lu_rb.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
            ratios_lu_rb_ignore=ratios_lu_rb_ignore.astype(np.float32))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids
