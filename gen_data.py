import argparse
from tqdm import tqdm

import numpy as np
from pathlib import Path
import cv2
import os
from utils.dataset import *



def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--train_num', type=int, default=10)
    ap.add_argument('--valid_num', type=int, default=2)
    ap.add_argument('--base_img', type=str, default=os.path.join('data', 'unlabeled', 'f_01324.png'))
    ap.add_argument('--labeled', type=str, default=os.path.join('data', 'labeled'))

    opt = vars(ap.parse_args())

    return opt


def main(opt):

    train_img_dir = make_dir('dataset', 'images', 'train')
    train_lbl_dir = make_dir('dataset', 'labels', 'train')
    valid_img_dir = make_dir('dataset', 'images', 'val')
    valid_lbl_dir = make_dir('dataset', 'labels', 'val')

    # ####################### LỌC LẤY GIỌT BÉ (notebook) #########################
    # Hướng 1: CLAHE trên toàn ảnh => Tệ
    # Hướng 2: CLAHE trên phần pixel màu xanh => Tệ
    # Hướng 3: HE trên cc => Tệ
    # Hướng 4: Không tiền xử lý nữa mà lấy cc luôn => OK

    print('[INFO] Preparing for image generation...')

    uimg = cv2.imread(opt['base_img'])
    uimg_gray = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
    uimg_mask = np.full_like(uimg_gray, 255)
    uimg_mask[np.where(uimg_gray == 255)] = 0

    uimg_orig = uimg.copy()

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(uimg_mask, connectivity=4)
    points = []
    class_name = []
    centers = []
    src_img = []

    for k in range(1, retval):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 60:
            loc = np.roll(np.where(labels == k), 1, axis=0).transpose().reshape(-1, 2)  # xy
            points.append(np.where(labels == k))  # yx
            class_name.append('small drop')
            centers.append(centroids[k])  # xy
            src_img.append(uimg_orig)

            # sau khi đánh nhãn xong thì xóa đi
            uimg[labels == k] = 255
            uimg_gray[labels == k] = 255
            uimg_mask[labels == k] = 0

    # Loại bỏ viền trước khi lọc tiếp (thủ công)
    loc = np.where(uimg_gray > 100)
    uimg[loc] = 255
    uimg_gray[loc] = 255
    uimg_mask[loc] = 0

    # Lọc tiếp lần nữa, y hệt bước trên, do sau khi bỏ viền sẽ có một số giọt tách ra
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(uimg_mask, connectivity=4)

    for k in range(1, retval):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 60:
            loc = np.roll(np.where(labels == k), 1, axis=0).transpose().reshape(-1, 2)
            points.append(np.where(labels == k))
            class_name.append('small drop')
            centers.append(centroids[k])
            src_img.append(uimg_orig)

            # sau khi đánh nhãn xong thì xóa đi
            uimg[labels == k] = 255
            uimg_gray[labels == k] = 255
            uimg_mask[labels == k] = 0

    # #################### LỌC LẤY GIỌT TO (notebook) ######################
    # Hướng 1: matchshape => Không thích :)
    # Hướng 2: blob detection => OK
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(uimg_mask, connectivity=4)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = False
    params.filterByInertia = False

    params.filterByCircularity = True
    params.minCircularity = 0.8

    params.filterByConvexity = True
    params.minConvexity = 0.96

    detector = cv2.SimpleBlobDetector_create(params)

    for k in range(1, retval):
        cc_mask = np.zeros_like(uimg_mask)
        cc_mask[labels == k] = 255

        loc = np.roll(np.where(labels == k), 1, axis=0).transpose().reshape(-1, 2)
        keypoints = detector.detect(255 - cc_mask)
        if len(keypoints) == 1:
            points.append(np.where(labels == k))
            class_name.append('large drop')
            centers.append(centroids[k])
            src_img.append(uimg_orig)

            # sau khi đánh nhãn xong thì xóa đi
            uimg[labels == k] = 255
            uimg_gray[labels == k] = 255
            uimg_mask[labels == k] = 0

    # ################# LẤY DETACHED LIGAMENT TỪ ẢNH ĐÃ GÁN NHÃN
    for lbl_path in Path(os.path.join(opt['labeled'], 'labels')).glob('*edit.txt'):  # f_01213-

        lbl_path = str(lbl_path)
        limg_path = lbl_path.replace('labels', 'images').replace('txt', 'png')

        boxes = process_txt(lbl_path)
        boxes = np.array([box[1:] + box[:1] for box in boxes])
        limg = cv2.imread(limg_path)
        boxes = yolo_to_coco(limg.shape[:2], boxes)
        boxes = xywh_to_xyxy(boxes)

        limg_gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        limg_mask = np.full_like(limg_gray, 255)
        limg_mask[limg_gray > 125] = 0

        for box in boxes:
            if int(box[-1]) == 2:  # neu la ligament
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = x1 - 4, y1 - 4, x2 + 4, y2 + 4
                cc_mask = np.zeros_like(limg_mask)
                cc_mask[y1:y2, x1:x2] = 255
                cc_mask[np.logical_not(np.logical_and(cc_mask == 255, limg_mask == 255))] = 0

                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(cc_mask, connectivity=4)
                k = \
                sorted([(k, stats[k, cv2.CC_STAT_AREA]) for k in range(1, retval)], key=lambda x: x[1], reverse=True)[
                    0][0]
                cc_mask[labels != k] = 0

                points.append(np.where(cc_mask == 255))
                class_name.append('deattached ligament')
                centers.append(centroids[k])
                src_img.append(limg)

    # ####################### DÁN CÁC GIỌT ĐÃ ĐÁNH NHÃN VÀO DÒNG CHÍNH ###############
    for mode, img_dir, lbl_dir, num in (('train', train_img_dir, train_lbl_dir, opt['train_num']),
                                        ('valid', valid_img_dir, valid_lbl_dir, opt['valid_num'])):
        start = len(os.listdir(img_dir))
        for i in tqdm(range(num), ascii=True, desc='[INFO] Generating for ' + mode):

            main = cv2.medianBlur(uimg, ksize=15)

            kernel = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]],
                              dtype=np.uint8)
            main_mask = cv2.morphologyEx(uimg_mask, cv2.MORPH_OPEN, kernel, iterations=8)

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(main_mask, connectivity=4)
            sizes = sorted([(k, stats[k, cv2.CC_STAT_AREA]) for k in range(1, retval)],
                           key=lambda x: x[1],
                           reverse=True)
            k_main = sizes[0][0]

            main_mask[labels != k_main] = 0
            main[labels != k_main] = 255
            main_loc = np.where(labels == k_main)

            syn_boxes = []

            # lấy các vị trí để dán attached ligament
            n_in = 10
            n_out = 10

            al_loc = np.where(cv2.erode(main_mask, kernel=kernel, iterations=15) == 255)

            rand_idx = np.random.choice([i for i in range(len(class_name)) if 'ligament' in class_name[i]],
                                        n_in + n_out)
            rand_loc = [(al_loc[0][i], al_loc[1][i]) for i in
                        np.random.choice([i for i in range(al_loc[0].shape[0]) if al_loc[1][i] > 1000], n_out)] + [
                           (al_loc[0][i], al_loc[1][i]) for i in
                           np.random.choice([i for i in range(al_loc[0].shape[0]) if al_loc[1][i] < 1000], n_in)]

            for idx, loc in zip(rand_idx, rand_loc):
                img = src_img[idx]
                mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
                mask[points[idx]] = 255

                transformed = transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

                main, main_mask, syn_loc = paste_seamless(img, main, mask, main_mask, (loc[1], loc[0]))
                x, y, w, h = cv2.boundingRect(syn_loc)
                syn_boxes.append([x, y, x + w, y + h, 4])

            # dán giọt
            # chưa áp dụng các kiểu xoay, zoom, làm méo ligament
            n = 200

            rand_idx = np.random.choice([i for i in range(len(class_name)) if class_name[i] != 'small drop'], n)

            n_in = int(0.5 * n)
            n_out = n - n_in
            rand_in_idx = np.random.choice([i for i in range(main_loc[0].shape[0]) if main_loc[1][i] < 1000], n_in)
            rand_out_idx = np.random.choice([i for i in range(len(centers))], n_out)
            rand_loc = [(main_loc[0][i], main_loc[1][i]) for i in rand_in_idx] + [
                (int(centers[i][1]), int(centers[i][0])) for i in rand_out_idx]  # yx

            for idx, loc in zip(rand_idx, rand_loc):

                img = src_img[idx]
                mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
                mask[points[idx]] = 255

                main, syn_loc = paste(img, main, mask, (loc[1], loc[0]))
                x, y, w, h = cv2.boundingRect(syn_loc)

                if 'drop' in class_name[idx]:
                    syn_boxes.append([x, y, x + w, y + h, 3])
                elif 'ligament' in class_name[idx]:
                    syn_boxes.append([x, y, x + w, y + h, 2])

            name = ('000000' + str(start + i))[-6:]
            cv2.imwrite(os.path.join(img_dir, name + '.jpg'), main)

            syn_boxes = coco_to_yolo(main.shape, xyxy_to_xywh(syn_boxes))
            buffer = []
            for box in syn_boxes:
                buffer.append("%d %.6f %.6f %.6f %.6f" % (box[4], box[0], box[1], box[2], box[3]))
            print('\n'.join(buffer), file=open(os.path.join(lbl_dir, name + '.txt'), 'w'))


if __name__ == '__main__':

    opt = parse_opt()

    main(opt)
