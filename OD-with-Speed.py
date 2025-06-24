# imports
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import math
from collections import deque
from tqdm import tqdm

from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from utils.augmentations import letterbox

import tensorflow as tf
import time
from pathlib import Path
from matplotlib import pyplot as plt
# %matplotlib inline

#Reuired Configuration
CFG_WEIGHTS = "./best.pb"
CFG_SOURCE = "./cut.mp4"
# CFG_SOURCE = "./car fast with offset.mp4"
CFG_IMAGESZ = 640
CFG_CONF = 0.25
CFG_SPEED_MODEL = "./deepspeed"

#Added Configuration
CFG_OUT = "./"
CFG_OUT_VIDEO = CFG_OUT + "out.avi"
CFG_OUT_VIDEO_ANNOTATED = CFG_OUT + "out_ant.avi"
CFG_GPU_ENABLE = False
CFG_STRIDES = 64
CFG_CONF_THRESH = 0.25
CFG_IOU_THRESH = 0.45
CFG_AGNOSTIC_NMS = False
CFG_MAX_DET = 10

#BBOX Configuration
CFG_LINE_THICKNESS = 3
CFG_NAMES = ['car', 'LP']

#DEBUG Configuration
CFG_SAMPLE_ORIGINAL_IMG = 'captured_original.png'
CFG_SAMPLE_MARKED_IMG = 'captured_updated.png'

#Early Path Checking
check_paths = [CFG_WEIGHTS, CFG_SOURCE]
for path in check_paths:
    if not Path(path).exists(): raise FileNotFoundError(f"file: {path} does not exist")

if not CFG_GPU_ENABLE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if not torch.cuda.is_available(): device = torch.device('cpu')
print('#### cpu selected successfully')

#load tensorflow model
def wrap_frozen_graph(gd, inputs, outputs):
    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs), tf.nest.map_structure(x.graph.as_graph_element, outputs))

graph_def = tf.Graph().as_graph_def()
graph_def.ParseFromString(open(CFG_WEIGHTS, 'rb').read())
frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")

print('#### model loaded successfully', frozen_func)

deepspeed = tf.keras.models.load_model(CFG_SPEED_MODEL)

"""corrected or new image sizes
basically stride should be multiple of stride
"""

# managing image resize, len==2 broken use the default for now
imgsz = check_img_size(CFG_IMAGESZ, s=CFG_STRIDES)
print('#### new image size', imgsz)

# Video Setting
FRAME = 0
CAP = cv2.VideoCapture(CFG_SOURCE)
video_out = Path(CFG_OUT_VIDEO)
print('>>>>>', video_out)
FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
WRITER_ORIGINAL = None
WRITER_ANNOTATED = None
def set_writer(size):
    return cv2.VideoWriter(str(CFG_OUT_VIDEO), cv2.VideoWriter_fourcc('M','J','P','G'), 30, size), cv2.VideoWriter(str(CFG_OUT_VIDEO_ANNOTATED), cv2.VideoWriter_fourcc('M','J','P','G'), 30, size)

def reset_cap(): return cv2.VideoCapture(CFG_SOURCE)

print('#### path to source resolved', Path(CFG_SOURCE).resolve().exists())
print('#### Number of frames found', FRAMES)

#Preprocessing pipeline
def preprocess_image(img):
    img = letterbox(img0, imgsz, stride=CFG_STRIDES, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    img = img[None]
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    return img

#Width thresholding definitions
WIDTH_THRESHOLD = 300

def plate_inside_car(car, plate):
    c2, c1 = car[0], car[1]
    p2, p1 = plate[0], plate[1]

    if ((c2[0] <= p1[0]) and (p1[0] <= c1[0])) and ((c2[0] <= p2[0]) and (p2[0] <= c1[0])) \
        and ((c2[1] <= p1[1]) and (p1[1] <= c1[1])) and ((c2[1] <= p2[1]) and (p2[1] <= c1[1])):
        return True
    return False

def point_difference(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    distance = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
    return distance

def point_speed(distance_moved):
    '''
    1000 m = 1 km
    1 px = 1.185 m = 0.001185 km
    1 F = 0.016666 sec = 0.00027777 hr
    '''
    pixel_distance = distance_moved * 0.001185 # km
    speed = pixel_distance / 0.00027777 # km/hr
    # print('speed', speed)
    return speed

def show(img, letter=True):
    if letter: img = letterbox(img, 640, stride=CFG_STRIDES, auto=False)[0]
    cv2.imshow('out', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

# Debugging
img_org = cv2.imread(CFG_SAMPLE_ORIGINAL_IMG)
# show(img_org)

img_cp = img_org.copy()
h, w = img_org.shape[:-1]
down_by_h = 500
p1, p2 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h) # 1st green
down_by_h = 50
p3, p4 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h) # 2nd green
down_by_h = 400
p5, p6 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h) # red
up_by_h = -250
p7, p8 = (0, int(h/2)+up_by_h), (w, int(h/2)+up_by_h) # gray
img_cp = cv2.line(img_cp, p1, p2, (0, 255, 0), 15)
img_cp = cv2.line(img_cp, p3, p4, (0, 255, 0), 15)
img_cp = cv2.line(img_cp, p5, p6, (0, 0, 255), 15)
img_cp = cv2.line(img_cp, p7, p8, (220, 220 ,220), 15)
# show(img_cp)

# Debugging pipeline
def reference_debug(img):
    img_cp = img.copy()
    h, w = img.shape[:-1]
    down_by_h = 500
    p1, p2 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h)
    down_by_h = 50
    p3, p4 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h)
    down_by_h = 400
    p5, p6 = (0, int(h/2)+down_by_h), (w, int(h/2)+down_by_h)
    up_by_h = -200
    p7, p8 = (0, int(h/2)+up_by_h), (w, int(h/2)+up_by_h)
    img_cp = cv2.line(img_cp, p1, p2, (0, 255, 0), 15)
    img_cp = cv2.line(img_cp, p3, p4, (0, 255, 0), 15)
    img_cp = cv2.line(img_cp, p5, p6, (0, 0, 255), 15)
    img_cp = cv2.line(img_cp, p7, p8, (220, 220 ,220), 15)
    return img_cp

def box_in_green_zone(x1, x2, sh):
    # print(x1, x2)
    h, w = sh[:-1]
    thu = int(h/2)+500
    thl = int(h/2)+50
    if x1[1] < thu and x1[1] > thl: return True
    return False

img_ = Path('captured_original.png')
img0_ = cv2.imread(str(img_))

imgg = img0_.copy()
img = letterbox(img0_, imgsz, stride=CFG_STRIDES, auto=False)[0]
img_o = img.copy()
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device)
img = img.float()
img /= 255.0
img = img[None]
imn = img.permute(0, 2, 3, 1).cpu().numpy()

pred = frozen_func(x=tf.constant(imn)).numpy()
pred[..., 0] *= imgsz # x
pred[..., 1] *= imgsz # y
pred[..., 2] *= imgsz # w
pred[..., 3] *= imgsz # h
pred = torch.tensor(pred)
pred = non_max_suppression(pred, CFG_CONF_THRESH, CFG_IOU_THRESH, None, CFG_AGNOSTIC_NMS, max_det=CFG_MAX_DET)

for i, det in enumerate(pred):
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        label = f'{CFG_NAMES[c]} {conf:.2f}'
        if c == 1:
            x1, y1, x2, y2 = [int(x) for x in xyxy]
            show(img_o[y1:y2, x1:x2], letter=True)

def save_image(image, where): cv2.imwrite(where, image)

# escape to break
# video is stored in out.avi
# CAP = reset_cap()
# Path(CFG_OUT_VIDEO_ANNOTATED).unlink(missing_ok=True)

previous_center_car = None
speeds = deque()
ROLLING_AVERAGE_LIMIT = 10
SPEED_FACTOR = 0.01
VALID_CAR_FRAMES = 1

SPEED_DETECTS = []
SPEED_SET = False
SPEED_DETECT_LIMIT = 10
S = None
RS = None
D = None
SM = None

PLATE_CAPT_COUNT = 5
PLATE_CAPTURED = False
with tqdm(total=FRAMES) as pbar:
    while FRAME != FRAMES:
        status, img0 = CAP.read()
        if not status:
            CAP.release()
            break

        if not WRITER_ORIGINAL:
            WRITER_ORIGINAL, WRITER_ANNOTATED = set_writer(img0.shape[0:2][::-1])

        # pre processing
        imgg = img0.copy()
        img = letterbox(img0, imgsz, stride=CFG_STRIDES, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        img = img[None]
        imn = img.permute(0, 2, 3, 1).cpu().numpy()

        pred = frozen_func(x=tf.constant(imn)).numpy()
        pred[..., 0] *= imgsz # x
        pred[..., 1] *= imgsz # y
        pred[..., 2] *= imgsz # w
        pred[..., 3] *= imgsz # h
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, CFG_CONF_THRESH, CFG_IOU_THRESH, None, CFG_AGNOSTIC_NMS, max_det=CFG_MAX_DET)
        for i, det in enumerate(pred):
            s, im0, frame = '', imgg.copy(), FRAME
            annotator = Annotator(im0, line_width=CFG_LINE_THICKNESS, example=str(CFG_NAMES))

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                valids_cars, valids_plate = [], []
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{CFG_NAMES[c]} {conf:.2f}'

                    # threshold code here
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    if WIDTH_THRESHOLD > 0:
                        BRP, TLP = p2, p1
                        width = BRP[0] - TLP[0]

                        if width >= WIDTH_THRESHOLD:
                            VALID_CAR_FRAMES += 1
                            if int(cls.numpy()) == 0: # class car
                                valids_cars.append([p1, p2, label])
                                hx, hy = int(BRP[0]/2), int(BRP[1]/2)

                                if previous_center_car:
                                    # ALL CAR SPEED HERE
                                    distance = point_difference(previous_center_car, (hx, hy))
                                    speed = point_speed(distance)
                                    speed += VALID_CAR_FRAMES * SPEED_FACTOR
                                    S = speed
                                    SM = deepspeed.predict([distance])[0][0]
                                    speeds.append(speed)
                                    if len(speeds) > ROLLING_AVERAGE_LIMIT:
                                        speeds.popleft()

                                    avg = np.array(speeds).mean()
                                    RS = avg
                                    annotator.box_label(xyxy, f"{avg:.2f} km/hr", color=colors(c, True))
                                    if box_in_green_zone(p1, p2, imgg.shape) and not SPEED_SET:
                                        SPEED_DETECTS.append(RS)
                                        if len(SPEED_DETECTS) == SPEED_DETECT_LIMIT:
                                            SPEED_SET = True
                                            D = np.array(SPEED_DETECTS).mean()
                                    else: pass
                                else:
                                    SPEED_SET = False # if car vanishes
                                    VALID_CAR_FRAMES = 1
                                    annotator.box_label(xyxy, f"processing km/hr", color=colors(c, True))

                                previous_center_car = (hx, hy)
                            else:
                                if int(cls.numpy()) == 1: # Number Plate
                                    valids_plate.append([p1, p2, label])

                for car in valids_cars:
                    for plate in valids_plate:
                        if plate_inside_car(car, plate):
                            if not PLATE_CAPTURED:
                                PLATE_CAPT_COUNT -= 1
                            if not PLATE_CAPT_COUNT and not PLATE_CAPTURED:
                                save_image(im0, './saved_frame.png')
                            annotator.box_label([plate[0][0], plate[0][1], plate[1][0], plate[1][1]], label, color=colors(c, True))

        im0 = annotator.result()
        if S: im0 = cv2.putText(im0, f"Instantaneous speed: {S:.2f} km/hr", (0, 100), 0, 3, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
        if S: im0 = cv2.putText(im0, f"Deep speed: {SM:.2f} km/hr", (0, 200), 0, 3, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
        if RS: im0 = cv2.putText(im0, f"Rolling avg speed: {RS:.2f} km/hr", (0, 300), 0, 3, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
        if D: im0 = cv2.putText(im0, f"Detected speed: {D:.2f} km/hr", (0, 400), 0, 3, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)

        im0 = reference_debug(im0)
        cv2.imshow('result', letterbox(im0, imgsz, stride=CFG_STRIDES, auto=False)[0])
        WRITER_ANNOTATED.write(im0)
        k = cv2.waitKey(10)
        if k == 27: break
        if k == ord('c'):
            cv2.imwrite('captured_original.png', img0)
            cv2.imwrite('captured_updated.png', im0)

        FRAME += 1
        pbar.update(1)

print(f'>>>> video saved to {CFG_OUT_VIDEO}')
cv2.destroyAllWindows()
