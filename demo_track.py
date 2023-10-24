from loguru import logger
import numpy as np

import cv2
from PIL import Image

from utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
import argparse
import os
import time
from yolo import YOLO

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="", help="path to images or video"
        "--path", default="", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(640, 640), type=tuple, help="test image size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=300, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def save_outputs(outputs, folder, save_name):
    sn = save_name.split('/')[-1].replace('.jpg', '.txt')
    # if not os.path.exists('yolov5_outputs'):
    #     os.mkdir('yolov5_outputs')

    sn = os.path.join('runs', folder, sn)
    # if not os.path.exists(os.path.join('yolov5_outputs', folder)):
    #     os.mkdir(os.path.join('yolov5_outputs', folder))
    # open or creat new file if dont exist
    with open(sn, 'w') as f:
        if outputs[0] is not None:
                for i in range(len(outputs[0])):
                    op = outputs[0][i].tolist()
                    for j in op:
                        f.write(str(j) + ' ')
                    f.write('\n')


def image_demo(predictor, vis_folder, path, current_time, save_result, save_name, test_size):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        outputs, img_info = predictor.inference(image_name, timer)
        save_outputs(outputs, save_name, image_name)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            # print('height:', img_info['height'], 'width:', img_info['width'])
            # print('test size:', exp.test_size)
            print('online_targets:', len(online_targets))
            # print(online_targets)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if save_result:
            save_folder = os.path.join(
                vis_folder, save_name
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            print("Save tracked image to {}".format(save_file_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)
        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    result_filename = os.path.join(vis_folder, os.path.basename(save_name + '.txt'))
    print("Save results to {}".format(result_filename))
    write_results(result_filename, results)


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_name = args.save_name
    save_folder = os.path.join(
        vis_folder, save_name
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, save_name + ".mp4")
    logger.info(f"video save_path is {save_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter('s02.mp4', fourcc, fps, size)

    tracker = BYTETracker(args, frame_rate=30)
    last_time = time.time()

    frame_id = 0
    results = []
    while True:
        last_time = time.time()
        if frame_id % 20 == 0:
            logger.info('Processing frame {}'.format(frame_id))
        ret_val, raw_frame = cap.read()
        raw_img = raw_frame.copy()
        if ret_val:
            frame = cv2.cvtColor(raw_frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            outputs, img_info = predictor.detect_obb_image(frame, raw_img)

            if len(outputs) > 0:  # [x1, y1, x2, y2]
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], args.tsize)
                online_tlwhs = []
                online_abtes = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    abt = t.abt
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_abtes.append(abt)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_abtes ,online_ids, frame_id=frame_id + 1,
                                          fps=1. / (time.time()-last_time))
            else:
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            cv2.imshow("online_im", online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main(args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    file_name = os.path.join('runs', '')
    os.makedirs(file_name, exist_ok=True)
    vis_folder = os.path.join(file_name, "track")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    model = YOLO()
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(model, vis_folder, args.path, current_time, args.save_result, args.save_name, args.tsize)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(model, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert args.demo in ["image", "video", "webcam"], "demo type not supported, only support [image, video, webcam]"
    main(args)