import os
import pickle
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from coco import COCODetection
from augmentation import BaseTransform

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')

dataset_root = "C:\\Users\hy211\PycharmProjects\datasets\MSCOCO"
output_dir = "C:\\Users\hy211\PycharmProjects\Sensitivity\output"
valid_save_dir = os.path.join(output_dir, 'ss_predict')
if not os.path.exists(valid_save_dir):
    os.makedirs(valid_save_dir)

valid_set = COCODetection(dataset_root, 'val', BaseTransform())
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
det_file = os.path.join(valid_save_dir, 'detections.pkl')

all_boxes = [[[] for _ in range(valid_set.__len__())] for _ in range(81)]
for n, sample in enumerate(valid_loader):
    sample = {key: value.to('cuda') for key, value in sample.items()}
    images = list(sample['image'])
    targets = sample['target']
    index = sample['index'].item()

    model.eval()
    pred = model(images)
    for i in range(pred[0]['boxes'].shape[0]):
        dets = torch.cat([pred[0]['boxes'], torch.unsqueeze(pred[0]['scores'], dim=1)], dim=1)
        cls = valid_set.coco_cat_idx_to_class_idx[pred[0]['labels'][i].item()]
        all_boxes[cls][index] = dets
    print('{}_image_finished'.format(n))

with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
APs, mAP = valid_set.evaluate_detections(all_boxes, valid_save_dir)

