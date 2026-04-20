# model distillation of TF++

This is a TransFuser-style training and inference workspace focused on the no-CARLA / distilled student pipeline.

## What is here

- `train.py` - command-line entry point for training the KD student model.
- `kd_trainer.py` - training loop and loss orchestration.
- `data.py` - dataset loading and preprocessing.
- `model.py`, `student_model.py`, `transfuser.py` - model definitions and wrappers.
- `center_net.py`, `gaussian_target.py` - bounding-box supervision utilities.
- `onnx_*`, `trt_infer.py`, `test*.py` - export, inference, and evaluation helpers.

## Setup

Install dependencies with the project requirements:

```bash
pip install -r requirements.txt
```

## Training

Example entry point:

```bash
python -m model_nocarla.train \
  --data_root /path/to/dataset \
  --config_path /path/to/config.json \
  --teacher_ckpt /path/to/teacher_checkpoint.pth \
  --output_dir /path/to/output \
  --experiment_name my_run
```

Use `--no_detect_boxes` if you want to disable the bounding-box head.

## Notes

- Keep datasets, checkpoints, logs, and exported models out of version control.
- If you add new generated artifacts, update `.gitignore` accordingly.