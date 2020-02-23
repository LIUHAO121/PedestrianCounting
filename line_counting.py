from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.config import cfg_mnet
from utils.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.load_model import load_model
from models.retinaface import RetinaFace
from utils.box_utils import decode
import time
import cv2
import torch
from sort import *
import os

writer = None

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def area(box):
    return abs((box[2] - box[0])*(box[3] - box[1]))

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-i', '--input_video', default='E:\\GraduateThesis\\code\\input\\tongjihu.mp4',type=str, help='Trained state_dict file path to open')
parser.add_argument('-o', '--output_video', default='E:\\GraduateThesis\\code\\output\\output_small.avi',type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_pic', default="E:\\GraduateThesis\\code\\output\\frame-{}.png",type=str, help='Trained state_dict file path to open')
parser.add_argument('-m', '--trained_model', default='E:\\seedland\\PedestrainCounting\\weights\\CrowdHumanP2_epoch_330.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')

args = parser.parse_args()
tracker = Sort()
memory = {}
counter = 0
line1 = [(1200, 1), (1200, 1079)]

save_root = 'E:\\GraduateThesis\\code\\output\\'
cap = cv2.VideoCapture(args.input_video)
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    resize = 1
    for c in os.listdir(save_root):
        os.remove(save_root + c)
    frameIndex = 0
    while 1:
        ret, img_raw = cap.read()
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        # (w,h,w,h)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)  # (h,w,c) -> (c,h,w)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        # priorbox的数量和大小根据每张图片的大小来确定
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

       # sort
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        real_time_people = 0

        if len(boxes) > 0:
            i = 0
            # plot box
            for box in boxes:
                (x1, y1) = (int(box[0]), int(box[1]))
                (x2, y2) = (int(box[2]), int(box[3]))
                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_raw, str(dets[:,-1][i]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                real_time_people += 1

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x3, y3) = (int(previous_box[0]), int(previous_box[1]))
                    (x4, y4) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                    p1 = (int(x3 + (x4 - x3) / 2), int(y3 + (y4 - y3) / 2))
                    cv2.line(img_raw, p0, p1, (255,0,0), 3)
                    # if (intersect(p0, p1, line1[0], line1[1]) or intersect(p0, p1, line2[0], line2[1]) or
                    #     intersect(p0, p1,line3[0],line3[1]) or intersect(p0, p1, line4[0], line4[1])) \
                    #         and LABELS[det_labels[i]] == 'person':
                    if intersect(p0, p1, line1[0], line1[1]):
                        counter += 1
                i += 1
        cv2.line(img_raw, line1[0], line1[1], (0, 255, 255), 5)
        cv2.putText(img_raw, 'HeadCount:' + str(int(counter)), (300, 100), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
        cv2.putText(img_raw, 'real_time: ' + str(real_time_people), (300, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)

        # saves image file
        cv2.imwrite(args.save_pic.format(frameIndex), img_raw)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output_video, fourcc, 30,
                                     (img_raw.shape[1], img_raw.shape[0]), True)

        # write the output frame to disk
        writer.write(img_raw)

        # increase frame index
        frameIndex += 1

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()