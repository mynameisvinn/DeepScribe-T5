# T1000
T1000 is a composable framework for training T5 models.

## Quickstart
### Local
`data_dir` refers to the folder containing `test_df.csv` and `train_df.csv`. `model` refers to the folder for checkpoints.
```bash
python scripts/train.py \
--data_dir data \
--output_path model
```