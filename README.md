# oyster-detection
An object detection and segmentation for Oysters. It uses [Detecron2](https://github.com/facebookresearch/detectron2) implementation.

## Config
Modify `oysterd/config.py` to modify the training and evaluation data folder, the folder containing the video to be used for inference, and the network structure of the model defined in [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). You can assign multiple network structures here.
### Parameters
- Config.folders['data']: **training and evaluation folder**
- Config.folders['infer']: inference folder
- Config.folders['output']: output folder
- Config.folders['weights']: folder where the trained network stored
- Config.config_file: a list of models defined in [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). You can select which model you want to train by passing its number in this list as the `<model_number>` argument to `python oysterd/train.py -i <model_number>`
- Config.resume: whether or not to resume/restart the training

## Train
Either run `python oysterd/train.py -i <model_number>` to train a single model `<model_number>` defined in `ysterd/config.py`, or run `sh run_trains.sh` for training multiple models.
The `-p` argument of `oysterd/train.py` determines whether or not predict the images in the evalution folder after training.

## Infer frames in a folder
Use `python oysterd/infer.py -i <model_number>` if you defined the video path in the `oysterd/config.py`. You can also pass the folder path by `-f` argument.

## Infer video frames
Use `python oysterd/infer_video.py -i <model_number>` if you defined the video path in the `oysterd/config.py`. You can also pass the video path by `-p` argument.

## Sample
<div align="center">
  <img src="https://github.com/bsadr/oyster-detection/blob/master/sample.png"/>
</div>
