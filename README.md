# shoeslace-detection
Recognizing untied and tied shoeslace using Tiny-YOLO v.2.

# Dataset
The dataset contains 245 vs. 230 annotations of untied and tied shoes pictures. The images are collected from Google, ImageNet and self-annotated.

# Steps:
1) Clone and install darkflow repo
2) Copy ```tiny-yolo-voc-2c.cfg``` to ```darkflow/cfg``` and replace ```darkflow/labels.txt``` with the label file
3) Copy simple_inference.py to folder darkflow
4) The training weights and checkpoints are stored at ```darkflow/ckpt``` (cannot upload due to exceeded bandwidth, pls contact me if you are interested in getting the weights)
5) Create a separate test folder to store test images/videos. Run this command:
```
cd darkflow/
python3 simple_inference.py --model [PATH_TO_MODEL] --load [NUMBER_OF_ITERATIONS] --threshold [THRESHOLD] --gpu [GPU_PERCENT] --test [PATH_TO_TEST_FOLDER]
```

The output result for example should look like this:

![test-video-yolo2](https://github.com/haantran96/shoeslace-detection/blob/master/test_video.gif)
