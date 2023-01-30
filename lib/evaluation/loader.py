import os

class WiderfaceLoader(object):
    """
    Load GT images in widerface format

    Parameters
    ----------
    path_to_label : path of the label file
    path_to_image : path of the folder of image files (cn0x)
    fname : name of the label file

    Properties:
    path_to_label
    path_to_image
    names: list of file names
    boxes: list of b-boxes

    -------
    Returns
    -------
    a wider parser
    """
    def __init__(self, path_to_label, fname):
        self.path_to_label = path_to_label

        fpath = os.path.join(path_to_label, fname)
        wf_file = open(fpath, 'r')
        lines = wf_file.readlines()
        imgnames = []
        imgboxes = []
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip('\n')
            if line[:2] == 'no':     # or endswith jpg
            # if a line starts with no_label: it is a start of a frame -> return name and boxes/[]
                i_name = line[9:]
                imgnames.append(i_name)
                if i + 1 < len(lines) and lines[i+1][:2] != 'no':          
                    amountOfBoxes = int(lines[i+1])
                    boxes = []
                    for temp in range(amountOfBoxes):
                        box = lines[i+2+temp].split(' ')[:4]
                        x1 = int(float(box[0]))
                        y1 = int(float(box[1]))
                        x2 = x1 + int(float(box[2]))
                        y2 = y1 + int(float(box[3]))
                        boxes.append([x1, y1, x2, y2])
                    imgboxes.append(boxes)
                else:
                    imgboxes.append([])
        self.names = imgnames
        self.boxes = imgboxes
        wf_file.close()