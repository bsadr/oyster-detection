# oyster-detection
An object detection and segmentation for Oysters. It uses [Detecron2](https://github.com/facebookresearch/detectron2) implementation.

## Config
Modify `oysterd/config.py` to modify the training and evalutation data folder, the folder containing the video to be used for inference, and the network structure of the model defined in [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). You can assign multiple network structures here.

## Train
Either run `python oysterd/train.py -i <model_number>` to train a single model `<model_number>` defined in `ysterd/config.py`, or run `sh run_trains.sh` for training multiple models.

## Infer video frames
Use `python oysterd/infer_video.py -i <model_number>` if you defined the video path in the `oysterd/config.py`. You can also pass the video path by `-v` argument.

## Sample
<div align="center">
  <img src="https://github.com/bsadr/oyster-detection/blob/master/sample.png"/>
</div>
