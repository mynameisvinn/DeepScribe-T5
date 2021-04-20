# T5
T5 is a sample training script that is invoked by EasyML/Sagemaker.

## Quickstart
```bash
python scripts/train.py \
--weights model \
--data_dir data \
--model_dir saved
```
`weights` refers to a folder containing `config.json` and `pytorch_model.bin`. `data_dir` refers to the folder containing `train_df.csv` and `test_df.csv`. `model_dir` refers to the folder that will be used to save updated model weights.
