import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2
import os


def process_txt(path):
    """
    Read .txt annotation file and convert to Python nested list.
    :param path: Python str, path to the .txt annotation file.
    :return: Python list of [label: int, c1: int, c2: int, c3: int, c4: int]
    """
    with open(path, 'r') as f:
        file_input = f.read()
    f.close()
    boxes = file_input.split('\n')[:-1]
    boxes = [box.split() for box in boxes]
    boxes = [[eval(i) for i in box] for box in boxes]
    return boxes


def yolo_to_coco(image_size, boxes):
    """
    Convert bounding box from YOLO format to COCO format.
    :param image_size: (height, width, ...)
    :param boxes: numpy array of shape (N, M>=4) in YOLO format [x, y, w, h, ...] normalized.
    :return: numpy array of shape (N, M) in COCO format [x, y, w, h, ...] in actual image size.
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    H, W = image_size[:2]
    new_boxes = np.copy(boxes).astype(np.int32)
    new_boxes[:, 0] = (boxes[:, 0] * W)
    new_boxes[:, 1] = (boxes[:, 1] * H)
    new_boxes[:, 2] = (boxes[:, 2] * W)
    new_boxes[:, 3] = (boxes[:, 3] * H)
    return new_boxes
    
    
def coco_to_yolo(image_size, boxes):
    """
    Convert bounding box from COCO format to YOLO format.
    :param image_size: (height, width, ...)
    :param boxes: numpy array of shape (N, M>=4) in YOLO format [x, y, w, h, ...] in actual image size.
    :return: numpy array of shape (N, M) in COCO format [x, y, w, h, ...] normalized.
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    H, W = image_size[:2]
    new_boxes = np.copy(boxes).astype(np.float32)
    new_boxes[:, 0] = (boxes[:, 0] / W)
    new_boxes[:, 1] = (boxes[:, 1] / H)
    new_boxes[:, 2] = (boxes[:, 2] / W)
    new_boxes[:, 3] = (boxes[:, 3] / H)
    return new_boxes
    

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [x, y, w, h, ...] format to [xmin , ymin, xmax, ymax, ...].
    :param boxes: boxes: numpy array of shape (N, M>=4).
    :return: numpy array of shape (N, M).
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    new_boxes = np.copy(boxes)
    new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2     ##############
    new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2     ##############
    new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2     ##############
    new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2     ##############
    return new_boxes


def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from [xmin , ymin, xmax, ymax, ...] format to [x, y, w, h, ...].
    :param boxes: boxes: numpy array of shape (N, M>=4).
    :return: numpy array of shape (N, M).
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    new_boxes = np.copy(boxes)
    new_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    new_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return new_boxes


def plot_image(image, cmap=None, **kwargs):
    """
    :param image: numpy array of shape (H, W, C) representing image in RGB color space.
    :param kwargs:
        boxes: numpy array - like of shape (N, 5) or (N, 6).
        font_size: int
        id_to_name_mapping: Python dictionary
        id_to_color_mapping: Python dictionary
        border_width: int
    :return:
    """
    # assert len(image.shape) == 3, f"Invalid shape for image, must be a 3-dim array, got {image.shape}"

    image = Image.fromarray(image)
    plotted_image = ImageDraw.Draw(image)

    if 'font_size' not in kwargs:
        kwargs['font_size'] = 9
    font = ImageFont.truetype('Arial.ttf', kwargs['font_size'])

    if 'border_width' not in kwargs:
        kwargs['border_width'] = 1

    if 'boxes' in kwargs:
        if not isinstance(kwargs['boxes'], np.ndarray):
            kwargs['boxes'] = np.array(kwargs['boxes'])

        assert kwargs['boxes'].shape[1] in [4, 5, 6], f"Invalid box shape: must be (N, M) where M must be in (4, 5, 6), got {kwargs['boxes'].shape}"
        for box in kwargs['boxes']:
            if box.shape[-1] > 4:
                class_id = int(box[-1])
            else:
                class_id = None
            if 'id_to_color_mapping' in kwargs:
                color = kwargs['id_to_color_mapping'][class_id]
            else:
                color = 'red'

            x1, y1, x2, y2 = box[:4]
            plotted_image.rectangle(
                ((x1, y1), (x2, y2)),
                outline=color,
                width=kwargs['border_width']
            )

            if class_id is None: class_name = ''
            else:
                if 'id_to_name_mapping' in kwargs: class_name = kwargs['id_to_name_mapping'][class_id]
                else: class_name = str(class_id)

                msg = ' ' + class_name + (f': {box[-2]:.2f} ' if kwargs['boxes'].shape[-1] == 6 else ' ')

                text_w, text_h = font.getsize(msg)
                plotted_image.rectangle(((x1, y1 - text_h), (x1 + text_w, y1)), fill=color, outline=color)
                plotted_image.text((x1, y1 - text_h), msg, fill='white', font=font)

    plt.imshow(image, cmap=cmap)
    return np.array(image)


def wipe(img, boxes):
    # cho ảnh và boxes của các giọt bắn trong ảnh.
    # tạm thời: đơn thuần xóa các boxes trong ảnh, để lại các khoảng trắng nhức mắt :-D
    assert len(boxes.shape) == 2, f"src_boxes must be a 2-D array. Got shape = {boxes.shape}"
    img = np.copy(img)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        img[y1:y2, x1:x2] = 255
    return img


def get_drops(img, boxes):
    # tạm thời cắt lấy các boxes hình chữ nhật theo trục, sau này chỉnh lại cho các dạng khác
    drops = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        drops.append(img[y1:y2, x1:x2])
    return drops


def augment(src_img, src_boxes, des_img, des_boxes):
    # tạm thời des_boxes = None, des_img là ảnh trắng.
    # src_boxes: xyxy
    assert len(src_boxes.shape) == 2, f"src_boxes must be a 2-D array. Got shape = {src_boxes.shape}"

    # tính tâm của các boxes trên ảnh gốc
    # lặp lại để lát nữa tiện tính toán
    des_centers = xyxy_to_xywh(src_boxes)
    des_centers[:, 2:4] = des_centers[:, :2]

    # shuffle các box (chỉ quan tâm tới kích thước của box)
    # lát nữa sẽ dán các mẩu ảnh đã shuffle vào các tâm cũ
    src_boxes = np.random.permutation(src_boxes)
    src_centers = xyxy_to_xywh(src_boxes)
    src_centers[:, 2:4] = src_centers[:, :2]

    if src_centers.shape[-1] > 4:
        des_centers[:, 4:] = src_boxes[:, 4:]

    # tọa độ của các box mới bằng tọa độ các box cũ + độ xê dịch của tâm
    des_boxes = src_boxes + (des_centers - src_centers)

    des_img = np.copy(des_img)
    for des_box, src_box in zip(des_boxes, src_boxes):
        # dán các box mới vào vị tâm cũ
        dx1, dy1, dx2, dy2 = des_box[:4]
        sx1, sy1, sx2, sy2 = src_box[:4]
        des_img[dy1:dy2, dx1:dx2] = src_img[sy1:sy2, sx1:sx2]

    return des_img, des_boxes
