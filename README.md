# T5
T5 is a NLU training template for EasyML.

## Parameters
Important arguments include:
* `model_dir`. SageMaker saves model weights in `model_dir` after completing training.
* `data_dir`. Training data is uploaded to `data_dir`.
* `checkpoint_path`. SagerMaker saves model weights in `checkpoint_path` so that training can resume after instance interruptions.