from loader import WiderfaceLoader

import matplotlib.pyplot as plt
import cv2
import os

## parsing (widerface) ==> annotated gt bboxes
# arg1: path to label
# arg2: path to images
# arg3: label file name (txt file)
# Wider = WIDER('arg1', 'arg2', 'wf.mat')

def load_gt(imgrootpath, cam_id):
    gt_labelpath = os.path.join("gt_loader", f"cn0{cam_id}_gt", "wider_face_split") 
    # gt_imgpath = os.path.join("data", f"cn0{cam_id}")

    gt_Wider = WiderfaceLoader(gt_labelpath, imgrootpath, 'wider_face_default_bbx_gt.txt')
      
    return gt_Wider

# show one example
imgrootpath = 'F:\ML3D-project\data\cn06'  # <-- input img folder path

gt_test = load_gt(imgrootpath, 6)

gt_imgnames = gt_test.names
gt_bbox = gt_test.boxes

gt_imgname_eg = gt_imgnames[0]  
gt_bbox_eg = gt_bbox[0]

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
