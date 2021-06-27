from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from tools.visualization import cv2_bboxes
import os
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    path = 'E:/HRSC2016/Test/AllImages'  # 做一个批量读取
    files = os.listdir(path)
    for ii, file in enumerate(files, 1):
        print(str(file))
        img = os.path.join(path, file)
        result = inference_detector(model, img)
    # show the results
        cv2_bboxes(result, img, score_thr=args.score_thr)
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
