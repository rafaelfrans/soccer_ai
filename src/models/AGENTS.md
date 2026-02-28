# Models Module

Wraps `ultralytics` YOLO training.

## `train_model()`

Thin wrapper around `model.train()`. All YOLO training params are passed through.

### Fine-Tuning Parameters

- `freeze` — Number of layers to freeze (24 = entire backbone, only detection head trains)
- `lr0` — Initial learning rate (use ~1e-4 for fine-tuning small datasets)
- `lrf` — Final LR factor (final_lr = lr0 * lrf, e.g. 0.01)

These are optional — omit them for standard training from scratch.

### Output

Results go to `{project}/{name}/` with `weights/best.pt` as the final model.
