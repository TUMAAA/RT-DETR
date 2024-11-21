```
# RT-DETR: Real-Time Detection Transformer

RT-DETR is a real-time object detection system based on the DETR (Detection Transformer) architecture.  It aims to achieve both high accuracy and fast inference speed, rivaling YOLO-series detectors.  RT-DETR introduces several innovations for efficient multi-scale feature processing and improved object query initialization, enabling real-time performance without sacrificing accuracy.  RT-DETRv2 builds upon RTDETR with further improvements using "bag-of-freebies" techniques to enhance the baseline performance.

This repository provides both PaddlePaddle and PyTorch implementations for RTDETR and RTDETRv2.

## Relevance

Real-time object detection is crucial for various applications like autonomous driving, video surveillance, and robotics.  Existing real-time detectors often face a trade-off between speed and accuracy.  RT-DETR addresses this challenge by leveraging the strengths of transformers while optimizing for efficiency, making it a valuable contribution to the field.

## Installation

### PyTorch

```bash
pip install -r requirements.txt
```

Compatible `torch` and `torchvision` versions:

| rtdetr | torch | torchvision |
|---|---|---|
| - | 2.4 | 0.19 |
| - | 2.2 | 0.17 |
| - | 2.1 | 0.16 |
| - | 2.0 | 0.15 |


### PaddlePaddle

```bash
pip install -r requirements.txt
```
PaddlePaddle 2.4.2 is required. See `requirements.txt` in the `rtdetr_paddle` directory.


## Data Preparation

### COCO

1. Download and extract the COCO 2017 dataset (images and annotations) to a directory of your choice.  The directory structure should be as follows:

```
path/to/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/    # train images
  val2017/      # val images
```

2. Update the dataset paths in the relevant config files:

**PyTorch:** `configs/dataset/coco_detection.yml`  (modify `img_folder` and `ann_file`)

**Paddle:** `configs/datasets/coco_detection.yml` (modify `dataset_dir`)


## Running the Code

### Training

**PyTorch (Single GPU):**

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

**PyTorch (Multiple GPUs):**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

**Paddle (Single GPU):**

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval
```

**Paddle (Multiple GPUs):**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

See the respective `README.md` files in the `rtdetr_pytorch` and `rtdetr_paddle` directories for more details on evaluation, inference, and exporting models.


## Training on Custom Data

1. **Prepare your data:** Organize your dataset in COCO format (or refer to the instructions in the original README for other formats).

2. **Update config files:** Modify the dataset paths, class names, and number of classes in the config files to reflect your custom dataset.

3. **Remapping categories:** If using COCO, set `remap_mscoco_category: False` in the config file. Update `mscoco_category2name` in `rtdetr_pytorch/src/data/coco/coco_dataset.py` if you want to remap categories.  Similar steps apply to the Paddle version.

4. **Finetuning:** For finetuning on custom data using pretrained weights, add the `-t path/to/checkpoint` flag to the training command. Refer to the `tools/README.md` file in each implementation directory for more details.



## Model Zoo and Benchmarks

Refer to the tables in the respective `README.md` files for pretrained model weights, performance metrics (AP and FPS), and corresponding config files.

## Citation

Refer to the Citation section in each `README.md` for the appropriate BibTeX entries to cite RTDETR and RTDETRv2.

```

