import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import albumentations as A



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


def background_from_color(H, W, color, offset=0):
    background = np.tile(color, H * W).reshape(H, W, -1).astype('uint8')
    background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    background[:, :, 2] = background[:, :, 2] + offset
    background = cv2.cvtColor(background, cv2.COLOR_HSV2BGR)
    color = background[0, 0, :]

    return background, color


def make_copy(*args):
    return [item.copy() for item in args]


def transform(image, mask):
    image, mask = make_copy(image, mask)
    H, W = image.shape[:2]
    image[mask == 0] = 0

    # cắt và xoay thẳng
    rect = cv2.minAreaRect(np.roll(np.where(mask == 255), 1, axis=0).transpose().reshape(-1, 2))
    (x, y), (w, h), alpha = rect
    if w < h:
        w, h = int(w), int(h)
        coord = np.array([[0, 0],
                          [0, h],
                          [w, h],
                          [w, 0]], dtype='float32')
    else:
        w, h = int(h), int(w)
        coord = np.array([[w, 0],
                          [0, 0],
                          [0, h],
                          [w, h]], dtype='float32')

    M = cv2.getPerspectiveTransform(cv2.boxPoints(rect), coord)
    image = cv2.warpPerspective(image, M, (w, h))

    # image = image[:-h//6]

    stretch = np.random.choice(range(5, 11))
    mag = np.random.choice([1.2, 1.4, 1.6])
    w, h = int(mag * w), int(mag * max(stretch * w, h))  # kéo dài ảnh tỉ lệ tối thiểu

    transformed = A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_CUBIC),
        A.Affine(scale=1.0, translate_percent=0, rotate=(-45, -145), shear=0, fit_output=True, p=1),
        A.PiecewiseAffine(p=1.0, nb_rows=5, nb_cols=5, mode='reflect'),
    ])(image=image)

    image = transformed['image']

    mask = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 40).astype('uint8') * 255

    return {'image': image, 'mask': mask}


def paste(src, des, src_mask, bottom_right):
    src, des, src_mask = make_copy(src, des, src_mask)

    if not isinstance(bottom_right, np.ndarray):
        bottom_right = np.array(bottom_right).reshape(1, 2)

    loc = np.roll(np.where(src_mask == 255), 1, axis=0).transpose().reshape(-1, 2)
    idx = np.argmax(loc[:, 0])

    new_loc = np.clip(loc - loc[idx] + bottom_right, 0, (des.shape[1] - 1, des.shape[0] - 1)).transpose()
    loc = loc.transpose()

    new_src = np.full_like(des, 255)
    new_src_mask = np.zeros(des.shape[:2], dtype='uint8')

    new_src[(new_loc[1], new_loc[0])] = src[(loc[1], loc[0])]
    new_src_mask[(new_loc[1], new_loc[0])] = 255

    des[new_src_mask == 255] = new_src[new_src_mask == 255]

    return des, new_loc.transpose()


def paste_seamless(src, des, src_mask, des_mask, bottom_right):
    src, des, src_mask, des_mask = make_copy(src, des, src_mask, des_mask)

    src[:, -src.shape[1] // 12:] = 0
    src_mask[:, -src.shape[1] // 12:] = 0

    if not isinstance(bottom_right, np.ndarray):
        bottom_right = np.array(bottom_right).reshape(1, 2)

    loc = np.roll(np.where(src_mask == 255), 1, axis=0).transpose().reshape(-1, 2)
    idx = np.argmax(loc[:, 0])

    new_loc = np.clip(loc - loc[idx] + bottom_right, 0, (des.shape[1] - 1, des.shape[0] - 1)).transpose()
    loc = loc.transpose()

    color = np.median(src[cv2.erode(src_mask, kernel=np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype='uint8'),
                                    iterations=3) == 255], axis=0)  # đang cấn ở đoạn này des[(new_loc[1], new_loc[0])]
    new_src, _ = background_from_color(des.shape[0], des.shape[1], color)
    # new_src = np.full_like(des, 255)
    new_src_mask = np.zeros_like(des_mask)

    new_src[(new_loc[1], new_loc[0])] = src[(loc[1], loc[0])]
    new_src_mask[(new_loc[1], new_loc[0])] = 255

    overlap_mask = cv2.bitwise_and(new_src_mask, des_mask)
    nonoverlap_mask = cv2.bitwise_xor(overlap_mask, new_src_mask)

    color = np.median(des[des_mask == 255], axis=0).astype('uint8')  # đang cấn ở đoạn này des_mask
    des[des_mask == 0] = color

    des_mask[nonoverlap_mask == 255] = 255

    new_src_mask = cv2.dilate(new_src_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)  # critical

    rect = cv2.boundingRect(new_src_mask)

    des = cv2.seamlessClone(new_src, des, new_src_mask, (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2),
                            cv2.NORMAL_CLONE)

    des[des_mask == 0] = 255

    return des, des_mask, new_loc.transpose()


def make_dir(*args):
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path