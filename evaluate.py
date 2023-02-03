import __init_paths__
from evaluation.loader import WiderfaceLoader
from torch.utils.data import DataLoader
from lib.data.test_data import TestData, collate
import utils.util as util
import os
import open3d as o3d
import json
from tqdm import tqdm

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

import os
import torch
# Data structures and functions for rendering
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
)
import numpy as np
import copy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def image_eval(pred, gt, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        # for h in range(overlaps.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list

IMAGE_H = 1536
IMAGE_W = 2048


def get_bboxes(image):
    """
    Returns the bounding boxes of each image
    [x_min, y_min, x_max, y_max]
    """

    img_b_channel = image[:, :, 0]
    x_min = None
    y_min = None
    x_max = None
    y_max = None
    for i in range(img_b_channel.shape[0]):
        row = np.where(img_b_channel[i, :] != 255)[0]
        if len(row) > 0:
            y_min = i
            break

    for i in range(img_b_channel.shape[0]):
        row = np.where(img_b_channel[-(i + 1), :] != 255)[0]
        if len(row) > 0:
            y_max = img_b_channel.shape[0] - i
            break

    for i in range(img_b_channel.shape[1]):
        column = np.where(img_b_channel[:, i] != 255)[0]
        if len(column) > 0:
            x_min = i
            break

    for i in range(img_b_channel.shape[1]):
        column = np.where(img_b_channel[:, -(i + 1)] != 255)[0]
        if len(column) > 0:
            x_max = img_b_channel.shape[1] - i
            break
    # print(x_min, y_min, x_max, y_max)
    if x_min == None or y_min == None or x_max == None or y_max == None:
        return np.array([None, None, None, None])
    # return np.array([x_min, y_min, x_max - x_min, y_max - y_min])

    return np.array([x_min, y_min, x_max, y_max])


def image_rendering(mesh, Rs, Ts, fls, pps, image_sizes):

    mesh = mesh.extend(len(image_sizes))

    cameras = PerspectiveCameras(R=Rs,
                                 T=Ts,
                                 focal_length=fls,
                                 principal_point=pps,
                                 image_size=image_sizes,
                                 in_ndc=False)

  
    raster_settings = RasterizationSettings(image_size=(image_sizes[0][0], image_sizes[0][1]),
                                            blur_radius=0.0,
                                            faces_per_pixel=1,
                                            cull_backfaces=True)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            # lights=lights
        )).to(device)

    images = renderer(mesh)

    return images


def convertMeshes2PT3DMeshes(mesh):
    """
    Converts a list of open3d meshes to a list of pytorch3d meshes

    :param meshes: List of open3d meshes
    :param texture_image: The image texture of the mesh
    :param trimesh_m_visual_uv: The UV values of the vertices of the mesh
    """

    faces = torch.tensor(np.array(mesh.triangles), dtype=torch.int64)[None, ...]
    verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)[None, ...]
    verts_rgb = torch.ones_like(torch.tensor(mesh.vertices), dtype=torch.float32)[None] # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))

    return Meshes(verts=[verts[0]], faces=[faces[0]], textures=textures).to(device)


def pt3d_camera_params(camera_params):
    """
    Returns the camera parameters in the form needed by pytorch3d

    :param camera_params: The camera parameters
    :return: The camera parameters in pytorch3d form
    """

    Rs = torch.empty(0)
    Ts = torch.empty(0)
    fls = torch.empty(0)
    pps = torch.empty(0)
    image_sizes = []

    for params in camera_params:

        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']

        world2color = np.zeros((4, 4))
        world2color[:3, :3] = params['R']
        world2color[:3, 3] = params['T'].flatten()

        F = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        world2color = F @ world2color

        R = torch.tensor(np.array([world2color[:3, :3].T]), dtype=torch.float32)
        T = torch.tensor(np.array([world2color[:3, 3]]), dtype=torch.float32)

        focal_length = torch.tensor([[fx, fy]], dtype=torch.float32)

        principal_point = torch.tensor([[cx, cy]], dtype=torch.float32)

        Rs = torch.cat((Rs, R), 0)
        Ts = torch.cat((Ts, T), 0)
        fls = torch.cat((fls, focal_length), 0)
        pps = torch.cat((pps, principal_point), 0)
        image_sizes.append([IMAGE_H, IMAGE_W])

    return Rs, Ts, fls, pps, image_sizes

def read_solution():
    solutions = {}
    path = os.path.join("data", "solutions", "filterreg")
    for f in os.listdir(path):
        frame_id = f[:-5]
        with open(os.path.join(path, f), "r") as file:
            solutions[frame_id] = json.load(file)
    return solutions


def blend_in(face_lists, _background_images):
    """
    Generates the image where all the faces are blended into the background image
    :param face_lists: 
    [
        [face_0_cam1, face_0_cam2, face_0_cam3, ...],
        [face_1_cam1, face_1_cam2, face_1_cam3, ...]
    ]
    :param background_image: the background images
    :param face_meta: contains the confidence scores etc.
    :param num_pixels: determines at what number of pixels the face shouldn't be displayed
    :param alpha_value: value for the alpha gradient mix of the poisson image editing
    """

    background_images = copy.deepcopy(_background_images)

    if len(face_lists) == 0:
        return background_images, np.empty((len(background_images), 0))

    face_lists = torch.stack((face_lists)).cpu().detach().numpy()

    bboxes = [[] for _ in range(len(background_images))]

    for img_i, background_image in enumerate(background_images):
        # We do not have a background image
        if background_image is None:
            continue
        for f_i, face_image in enumerate(face_lists[:, img_i]):
            # convert to cv2 standards
            face_image = cv2.cvtColor(face_image * 255, cv2.COLOR_RGBA2BGRA)

            background_images[img_i] = trivial_overlapping(face_image, background_image)

            x_min, y_min, x_max, y_max = get_bboxes(face_image)

            # Adding it to the bbox array without the buffer
            if x_min is not None:
                bboxes[img_i].append(
                    np.array([x_min, y_min, x_max - x_min, y_max - y_min]))

    return background_images, np.array(bboxes, dtype=object)


def trivial_overlapping(source, target):
    color = np.array(source / 255, dtype=np.float32)
    raw_img = target

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

    output_img = 255 - ((255 - color[:, :, :-1]) * valid_mask + (1 - valid_mask) * raw_img)

    output_img = (output_img * 255).astype(np.uint8)

    return output_img


camera_list = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]
def render(data_loader):

    results = {
        "cn01" : [0, 0, 0],
        "cn02" : [0, 0, 0], 
        "cn03" : [0, 0, 0],
        "cn04" : [0, 0, 0],
        "cn05" : [0, 0, 0],
        "cn06" : [0, 0, 0]
    }

    identity = False

    gt = load_gt()
    solutions = read_solution()

    camera_params = util.load_camera_params(os.path.join("data", "trial"))

    Rs, Ts, fls, pps, image_sizes = pt3d_camera_params(camera_params)

    for i, data_item in tqdm(enumerate(data_loader), total=len(data_loader)):
        raw_images = []
        face_lists = []
        frame_id = data_item["frame_id"][0]
        mesh_ids = [x for x in data_item["ids"]]
        meshes = [o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vert[0]), triangles=o3d.utility.Vector3iVector(triang[0])) for vert, triang in zip(data_item["face_vert"], data_item["face_triang"]) if len(vert) != 0 and len(triang) != 0]

        for cam_id, cam in enumerate(camera_list):
            file_id = str(frame_id).zfill(10)
            fpath = os.path.join("data", "trial", cam, f"{file_id}_color.jpg")
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            raw_images.append(img)

        for i in range(len(meshes)):
            mesh = meshes[i]
            mesh_id = mesh_ids[i][0]

            if not identity:
                try:
                    # transformed_verts = (np.array(mesh.vertices) - np.array(solutions[str(frame_id)][str(mesh_id)]["trans"])) @ np.array(solutions[str(frame_id)][str(mesh_id)]["rot"])
                    # mesh.vertices = o3d.utility.Vector3dVector(transformed_verts)
                    tf_matrix = np.eye(4)
                    tf_matrix[:3, :3] = solutions[str(frame_id)][str(mesh_id)]["rot"]
                    tf_matrix[:3, 3] = np.array(solutions[str(frame_id)][str(mesh_id)]["trans"]) 
                    mesh = mesh.transform(tf_matrix)
                except KeyError:
                    pass

            pt3d_mesh = convertMeshes2PT3DMeshes(mesh)
            face_images = image_rendering(pt3d_mesh, Rs, Ts, fls, pps, image_sizes)
            face_list = torch.stack([
                face_images[j] for j in range(len(face_images))
            ])

            face_lists.append((face_list))
        # List of pt3d_meshes
        images, bboxes_frame = blend_in(face_lists,
                                        raw_images)
        for j, image in enumerate(images):
            if image is None:
                continue
            bbox_image = copy.deepcopy(image)
            for x_min, y_min, width, height in bboxes_frame[j]:
                if x_min is None:
                    continue
                bbox_image = cv2.rectangle(bbox_image, (x_min, y_min),
                                            (x_min + width, y_min + height), (0, 0, 255), 1)
            
            os.makedirs(f"output/filterreg/cn0{j+1}", exist_ok=True)
            cv2.imwrite(f"output/filterreg/cn0{j+1}/{str(frame_id).zfill(10)}_color.jpg", bbox_image)

        for ij, cam_ij in enumerate(camera_list):
            pred = np.array(bboxes_frame[ij])
            target = np.array(gt[cam_ij][f"{str(frame_id).zfill(10)}_color.jpg"])

            results[cam_ij][1] += len(pred)
            results[cam_ij][2] += len(target)
            if len(pred) == 0 or len(target) == 0:
                pred_recall = [0]
            else:
                pred_recall, proposal_list = image_eval(pred, target, 0.4)

            detected_face = pred_recall[-1]  # image_eval_2(pred_info, gt_boxes, iou_thresh)

            results[cam_ij][0] += detected_face


    print(results)
    for cam, v in results.items():
        print(f"{cam}: recall: {np.around(v[0] / v[2] * 100, 2)}, precision: {np.around(v[0] / v[1] * 100, 2)}")


def write_preds(creator, bboxes):
    """
    Writes down the predictions which were calculated by the render method

    :param config: the configs
    :param bboxes: the bounding boxes calculated by the render method
    """

    camera_outputs = dict()  # {<camera> : <output_str>}
    output_str = ""
    # Create new predictions directory if one already persists
    next_try = "self"
    if os.path.exists(os.path.join(creator.output_dir, "holistic_3d", "predictions", next_try)):
        t = 1
        next_try = f"self_{t}"
        while (os.path.exists(os.path.join(creator.output_dir, "holistic_3d", "predictions", next_try))):
            t += 1
    output_dir = creator.prepare_output_dirs(prefix='holistic_3d/predictions', dataDirs=[next_try])[0]

    for frame_id, cameras_and_meshes in bboxes.items():
        for cam_num, meshes_and_bbox in enumerate(cameras_and_meshes):
            cam_str = f"cn0{cam_num + 1}"
            file_str = f"{str(frame_id).zfill(10)}_color.jpg" + "\n"

            content_bboxes = ''
            num_faces = 0
            for [x_min, y_min, x_max, y_max, score] in meshes_and_bbox:
                if x_min != None:
                    num_faces += 1
                    content_bboxes += str(int(x_min)) + ' ' + str(int(y_min)) + ' ' + str(
                        int(x_max)) + ' ' + str(int(y_max)) + ' ' + str(score) + '\n'
            content = str(num_faces) + '\n' + content_bboxes

            output_str = file_str + content
            if cam_str in camera_outputs:
                camera_outputs[cam_str] += output_str
            else:
                camera_outputs[cam_str] = output_str

    for cam_id, output_string in camera_outputs.items():
        outFile = open(os.path.join(output_dir, f"{cam_id}.txt"), 'w')
        outFile.write(output_string)

if __name__=='__main__':
    test_loader = DataLoader(TestData(1024, load=["face"]), num_workers=1, collate_fn=collate)
    render(test_loader)
    