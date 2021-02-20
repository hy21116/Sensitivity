import os
import cv2
import torch
import json
import pickle
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCODetection(data.Dataset):
    def __init__(self, root, val, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.cache_path = os.path.join(self.root, 'cache')
        self.annotations = list()

        anno_path = os.path.join(self.root, 'annotations', 'instances_{}2017.json'.format(val))
        self._COCO = COCO(anno_path)
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_class = len(self.classes)
        class_to_idx = dict(zip(self.classes, range(self.num_class)))
        self.class_to_coco_cat_idx = dict(zip([c['name'] for c in cats], self._COCO.getCatIds()))
        self.coco_cat_idx_to_class_idx = dict([(self.class_to_coco_cat_idx[cls], class_to_idx[cls])
                                               for cls in self.classes[1:]])
        self.indexes = self._COCO.getImgIds()
        self.img_paths = [os.path.join(self.root, 'images', '{}2017'.format(val), str(i).zfill(12)+'.jpg')
                     for i in self.indexes]
        self.annotations.extend(self.load_annotations(val, self.indexes))

        # self.annotation_from_index(397133)
        # print('for_debug!')

    def load_annotations(self, val, indexes):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        ## TODO The following code is for what?
        cache_file = os.path.join(self.cache_path, '{}2017'.format(val)+'update_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            return roidb

        annotations = [self.annotation_from_index(i) for i in indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(annotations,fid, pickle.HIGHEST_PROTOCOL)
        return annotations

    def annotation_from_index(self, index):
        img_ann = self._COCO.loadImgs(index)[0]
        width = img_ann['width']
        height = img_ann['height']

        img_ann_ids_list = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        img_objs_ann_list = self._COCO.loadAnns(img_ann_ids_list)
        valid_objs = []
        for obj in img_objs_ann_list:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width-1, x1+np.max((0, obj['bbox'][2]))))
            y2 = np.min((height-1, y1+np.max((0, obj['bbox'][3]))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_box'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        num_objs = len(valid_objs)

        boxes_info = np.zeros((num_objs, 5))
        for i, obj in enumerate(valid_objs):
            cls = self.coco_cat_idx_to_class_idx[obj['category_id']]
            # cls = obj['category_id']
            boxes_info[i, 0:4] = obj['clean_box']
            boxes_info[i, 4] = cls
        return boxes_info

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        target = self.annotations[index]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        img_ann_ids_list = self._COCO.getAnnIds(self.indexes[index])
        img_objs_ann_list = self._COCO.loadAnns(img_ann_ids_list)

        target = np.array(target)
        target[:, 0] /= width
        target[:, 2] /= width
        target[:, 1] /= height
        target[:, 3] /= height

        sample = {'image': img, 'target': target, 'height': height, 'width': width}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)
        sample['target'] = torch.from_numpy(sample['target']).float()
        sample['index'] = torch.tensor([index])
        return sample

    def __len__(self):
        return len(self.img_paths)

    def coco_results_one_category(self, boxes, cat_id):
        results = []
        for i, img_idx in enumerate(self.indexes):
            cls_img_dets = boxes[i]
            if cls_img_dets == []:
                continue
            # cls_img_dets = cls_img_dets.astype(np.float)
            scores = cls_img_dets[:, -1]
            x = cls_img_dets[:, 0]
            y = cls_img_dets[:, 1]
            w = cls_img_dets[:, 2] - x
            h = cls_img_dets[:, 3] - y
            results.extend(
                [{'image_id': img_idx,
                  'category_id': cat_id,
                  'bbox': [x[k], y[k], w[k], h[k]],
                  'score': scores[k]} for k in range(cls_img_dets.shape[0])]
            )
        return results

    def write_coco_results_json_file(self, all_boxes, json_file):
        results = []
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            results.extend(self.coco_results_one_category(all_boxes[i], self.class_to_coco_cat_idx[cls]))

        with open(json_file, 'w') as fid:
            json.dump(results, fid)

    # def do_detection_eval(self, json_file, valid_save_dir):


    def evaluate_detections(self, all_boxes, valid_save_dir):
        json_file = os.path.join(valid_save_dir, ('detections_val2017_results.json'))
        self.write_coco_results_json_file(all_boxes, json_file)
        APs, mAP = self.do_detection_eval(json_file, valid_save_dir)