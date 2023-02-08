from loader import WiderfaceLoader

import matplotlib.pyplot as plt
import cv2
import os

## parsing (widerface) ==> annotated gt bboxes
# arg1: path to label
# arg2: path to images
# arg3: label file name (txt file)
# Wider = WIDER('arg1', 'arg2', 'wf.mat')

def load_gt():
    gt = {}
    for i in range(6):
        cam_dict = {}
        gt[f"cn0{i + 1}"] = cam_dict
        gt_labelpath = os.path.join("data", "gt", f"cn0{i + 1}", "wider_face_split") 

        gt_Wider = WiderfaceLoader(gt_labelpath, 'wider_face_default_bbx_gt.txt')
        for fname, boxes in zip(gt_Wider.names, gt_Wider.boxes):
            cam_dict[fname] = boxes
        
    return gt

# show one example
imgrootpath = "data/trial/cn06"  # <-- input img folder path

gt_test = load_gt()["cn06"]

gt_imgnames = list(gt_test.keys())
gt_bbox = list(gt_test.values())

gt_imgname_eg = gt_imgnames[0]  
gt_bbox_eg = gt_bbox[0]
print(os.path.join(imgrootpath, gt_imgname_eg))

im = cv2.imread(os.path.join(imgrootpath, gt_imgname_eg))
im = im[:, :, (2, 1, 0)]
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(im, aspect='equal')

for gt_bbox_indv in gt_bbox_eg:
    ax.add_patch(
        plt.Rectangle((gt_bbox_indv[0], gt_bbox_indv[1]),
                    gt_bbox_indv[2] - gt_bbox_indv[0],
                    gt_bbox_indv[3] - gt_bbox_indv[1], fill=False,
                    edgecolor='red', linewidth=3.5)
        )
        
plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()




# DELETE THIS FILE IT IS FOR DEBUGGING ONLY