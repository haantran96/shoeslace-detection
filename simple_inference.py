import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import argparse


def inference(model,load,threshold,gpu,test_path):
    params = {
        'model': model,
        'load': load,
        'threshold': threshold,
        'gpu': gpu
    }
    gif = test_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tfnet = TFNet(params)
    tfnet.load_from_ckpt()

    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
    capture = cv2.VideoCapture(gif)
    out = cv2.VideoWriter('/home/an/shoeslace/test/test_video.mp4',fourcc, 20.0, (int(capture.get(3)),int(capture.get(4))))

    while (capture.isOpened()):
        start = time.time()
        ret, frame = capture.read()

        if ret:
            results = tfnet.return_predict(frame)
            for c,r in zip(colors,results):
                top_left = (r["topleft"]["x"], r["topleft"]["y"])
                bottom_right = (r["bottomright"]["x"], r["bottomright"]["y"])
                label = r["label"]
                confidence = r['confidence']

                print(top_left,bottom_right,label,confidence)
                frame = cv2.rectangle(frame,top_left,bottom_right,c,thickness=2)
                frame = cv2.putText(frame, label+" "+str(confidence), top_left, cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=0.5,color = c,thickness=1)
                out.write(frame)
                print("WRITING FRAMES")

            # Insert FPS/quit text and show image
            fps = "{:.0f} FPS".format(1 / (time.time() - start))
            cv2.imshow('Frame', frame)
            frame = cv2.putText(frame, fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(0, 0, 0))
            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            capture.release()
            out.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO2 inference!')
    parser.add_argument("--model", default="cfg/tiny-yolo-voc-2c.cfg", help="directory to the model")
    parser.add_argument("--load", default=-1, help="numper of loads")
    parser.add_argument("--threshold", default=0.1, help="threshold to select")
    parser.add_argument("--gpu",default=1, help="Percent of used GPU")
    parser.add_argument("--test",default="/home/an/shoeslace/test/test_video_1.mp4", help="Directory to test files")


    args = parser.parse_args()
    inference(args.model, int(args.load), float(args.threshold), float(args.gpu),args.test)

