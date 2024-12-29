
import sys
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = Path(FILE.parents[0], "yolov5")  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow,
                           check_requirements, colorstr, cv2, increment_path,
                           non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class yolov5_detect():

    @torch.no_grad()
    def __init__(self,
                 weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                 source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                 data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                 detect_mode='auto',  # detect mode, 'auto' or 'frame_by_frame'
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=False,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project=ROOT / 'runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 fbf_output_name="output",  # output name for frame_by_frame mode
                 fbf_output_fps: int = 60,  # output fps for frame_by_frame mode
                 fbf_get_frame_annotator=False,  # get frame annotator for frame_by_frame mode
                 fbf_close_logger_output=False,  # close logger output for frame_by_frame mode
                 ) -> None:
        self.detect_mode = detect_mode
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.fbf_output_name = fbf_output_name
        self.fbf_output_fps = fbf_output_fps
        self.fbf_get_frame_annotator = fbf_get_frame_annotator
        self.fbf_close_logger_output = fbf_close_logger_output

        # Start Init
        source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download
        self.source = source

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=self.imgsz,
                                       stride=self.stride, auto=self.pt)
            self.bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            self.bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * self.bs, [None] * self.bs

        if self.detect_mode == 'frame_by_frame':
            self.count = 0
            self.dataset.mode = "video"
            self.detect_frame_by_frame_init()

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup

    def detect_frame_by_frame_init(self):
        self.seen, self.windows, self.dt = 0, [], [0.0, 0.0, 0.0]
        self.path = self.fbf_output_name
        self.vid_cap = None

    def run(self, frame: np.array = None):
        """Start detection on the given image(s) or video(s)

        Args:
            frame (`np.array`, optional): give frame when using frame_by_fram mode. Defaults to None.
        """
        if self.detect_mode == 'auto':
            self.detect_auto()
            self.print_results()
        elif self.detect_mode == 'frame_by_frame':
            self.detect_frame_by_frame(frame)

    def detect_frame_by_frame(self, frame: np.array):
        """Detect frame by frame"""
        # Preprocess image
        self.im0s, self.s = frame, ''

        t1 = time_sync()
        # Padded resize
        self.im = letterbox(self.im0s, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        self.im = self.im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        self.im = np.ascontiguousarray(self.im)

        # to Tensor
        self.im = torch.from_numpy(self.im).to(self.device)
        self.im = self.im.half() if self.model.fp16 else self.im.float()  # uint8 to fp16/32
        self.im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(self.im.shape) == 3:
            self.im = self.im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        self.visualize = increment_path(self.save_dir / Path(self.path).stem,
                                        mkdir=True) if self.visualize else False
        self.pred = self.model(self.im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        self.dt[1] += t3 - t2
        self.pred_time = t3 - t2

        # NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres,
                                        self.classes, self.agnostic_nms, max_det=self.max_det)

        t4 = time_sync()
        self.dt[2] += t4 - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        self.precess_predictions()
        if not self.fbf_close_logger_output:
            t = ((t2 - t1) * 1E3, (t3 - t2) * 1E3, (t4 - t3) * 1E3)
            LOGGER.info(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

    def detect_auto(self):
        """Run inference on the input image(s)"""
        self.seen, self.windows, self.dt = 0, [], [0.0, 0.0, 0.0]
        for self.path, self.im, self.im0s, self.vid_cap, self.s in self.dataset:
            t1 = time_sync()
            self.im = torch.from_numpy(self.im).to(self.device)
            self.im = self.im.half() if self.model.fp16 else self.im.float()  # uint8 to fp16/32
            self.im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(self.im.shape) == 3:
                self.im = self.im[None]  # expand for batch dim
            t2 = time_sync()
            self.dt[0] += t2 - t1

            # Inference
            self.pred = self.model(self.im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            self.dt[1] += t3 - t2
            self.pred_time = t3 - t2

            # NMS
            self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres,
                                            self.classes, self.agnostic_nms, max_det=self.max_det)
            self.dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            self.precess_predictions()

    def precess_predictions(self):
        # Process predictions
        for i, det in enumerate(self.pred):  # per image
            if self.webcam:  # batch_size >= 1
                p, im0, frame = self.path[i], self.im0s[i].copy(), self.dataset.count
                self.s += f'{i}: '
            elif self.detect_mode == 'frame_by_frame':
                p, im0, frame = self.path, self.im0s.copy(), self.seen
            else:
                p, im0, frame = self.path, self.im0s.copy(), getattr(self.dataset, 'frame', 0)
            self.seen += 1

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + \
                ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
            self.s += '%gx%g ' % self.im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.im.shape[2:], det[:, :4], im0.shape).round()

                # save pred result to output
                pred_np = det.clone().cpu().numpy()
                self.pred_np = pred_np

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls,
                                                                          *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.save_crop or self.view_img or self.fbf_get_frame_annotator:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (
                            self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        self.im0_result = im0
                    if self.save_crop:
                        save_one_box(xyxy, imc, file=self.save_dir / 'crops' /
                                     self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                if p not in self.windows:
                    self.windows.append(p)
                    # allow window resize (Linux)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if self.vid_path[i] != save_path:  # new video
                        self.vid_path[i] = save_path
                        if isinstance(self.vid_writer[i], cv2.VideoWriter):
                            self.vid_writer[i].release()  # release previous video writer
                        if self.vid_cap:  # video
                            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        elif self.detect_mode == 'frame_by_frame':
                            fps, w, h = self.fbf_output_fps, im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        self.vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    self.vid_writer[i].write(im0)

        # Print time (inference-only)
        if self.detect_mode != 'frame_by_frame':
            LOGGER.info(f'{self.s}Done. ({self.pred_time:.3f}s)')

    def print_results(self):
        # Print results
        t = tuple(x / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)