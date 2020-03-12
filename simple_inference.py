import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

params = {
    'model': 'cfg/tiny-yolo-voc-2c.cfg',
    'load': 2000,
    'threadhold': 0.15,
    'gpu': 1
}

gif = "/shoeslace/test/test_video_1.mp4"

tfnet = TFNet(params)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
capture = cv2.VideoCapture(gif)
print(capture.isOpened())

while (capture.isOpened()):
    start = time.time()
    ret, frame = capture.read()

    if ret:
        results = tfnet.return_predict(frame)
        for c,r in zip(colors,results):
            top_left = (r["topleft"]["x"], r["topleft"]["y"])
            bottom_right = (r["bottomright"]["x"], r["bottomright"]["y"])
            label = r["label"]

            frame = cv2.rectangle(frame,top_left,bottom_right,c,thickness=10)
            frame = cv2.putText(frame, label, top_left, cv2.FONT_HERSHEY_COMPLEX,
                                1,(0,0,0),thickness=2)


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
        cv2.destroyAllWindows()
        break