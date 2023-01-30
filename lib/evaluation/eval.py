def cal_IOU_for_1_box(gt_1bbox, pred_1bbox):
    """
    Calculate IOU score for 1 pair of gt bbox and pred bbox
    --------
    Inputs: 1 gt bbox & 1 pred bbox
    Return: 1 IOU score
    """

    # areas of both bboxes:
    gt_area = (gt_1bbox[2] - gt_1bbox[0] + 1) * (gt_1bbox[3] - gt_1bbox[1] + 1)
    pred_area = (pred_1bbox[2] - pred_1bbox[0] + 1) * (pred_1bbox[3] - pred_1bbox[1] + 1)

    # get x,y coordinates of the Intersection
    Ix1 = max(gt_1bbox[0], pred_1bbox[0])
    Iy1 = max(gt_1bbox[1], pred_1bbox[1])
    Ix2 = min(gt_1bbox[2], pred_1bbox[2])
    Iy2 = min(gt_1bbox[3], pred_1bbox[3])

    # intersection area
    intersection = (Ix2 - Ix1 + 1) * (Iy2 - Iy1 + 1)
    iou = intersection / float(gt_area + pred_area - intersection + 1e-16) # avoid divided by 0
    return iou



def eval_IOU(gt_bbox, pred_bbox):
    """
    Calculate IOU scores for 2 lists of gt bboxes and pred bboxes
    --------
    Inputs: 2 lists of gt bbox & pred bbox
    Return: a list of IOU scores
        If 2 empty boxes: return []
        If one box is empty, the other is not: return 0
        If both having 2 boxes: pick a larger combination
        If 1 box versus 2 boxes: pick a larger one

    """

    assert len(gt_bbox) == len(pred_bbox), 'Unequal sequences of b-boxes.'

    num_boxes = len(gt_bbox)
    IOU = []
    for n in num_boxes:
        len_gt = len(gt_bbox[n])
        len_pred = len(pred_bbox[n])
        if len_gt == 0 and len_pred == 0 :
            IOU.append([])
        elif len_gt == 0 or len_pred == 0 :
            iou_score = 0
            IOU.append(iou_score)
        else:
            if len_gt == len_pred:    # 2 boxes gt & pred
                opt1 = []
                opt2 = []
                for l in len_gt:
                    temp = cal_IOU_for_1_box(gt_bbox[n][l], pred_bbox[n][l])
                    opt1.append(temp)
                for l in len_gt:
                    temp = cal_IOU_for_1_box(gt_bbox[n][l], pred_bbox[n][1-l])
                    opt2.append(temp)
                if sum(opt1) >= sum(opt2):
                    iou_score = opt1
                else:
                    iou_score = opt2
                IOU.append(iou_score)
            elif len_gt > len_pred:   # 2 gt boxes, 1 pred box
                iou_score = []
                for l in len_gt:
                    temp = cal_IOU_for_1_box(gt_bbox[n][l], pred_bbox[n])
                    iou_score.append(temp)
                iou_score = max(iou_score)
                IOU.append(iou_score)
            else:                     # 2 pred boxes, 1 gt box
                iou_score = []
                for l in len_pred:
                    temp = cal_IOU_for_1_box(gt_bbox[n], pred_bbox[n][l])
                    iou_score.append(temp)
                iou_score = max(iou_score)
                IOU.append(iou_score)
    return IOU


    
def evaluation(type, gt_bbox, IOU, IOU_threshold = 0.5):

    assert len(gt_bbox) == len(IOU), 'Unequal sequences.'

    tp = 0
    fp = 0
    fn = 0
    num_gt = 0

    for i, gt in enumerate(gt_bbox):
        if len(gt) != 0:
            num_gt += len(gt)
            for n in len(gt):
                if IOU[n][i] > IOU_threshold:
                    tp += 1
            else:
                fp += 1

    fn = num_gt - tp

    if type == 'precision':
        return tp / (tp + fp + 1e-16)
    elif type == 'recall':       
        return tp / (tp + fn)
    elif type == 'F1':
        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn)
        if precision == 0 or recall == 0 :
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall + 1e-16)
        return F1