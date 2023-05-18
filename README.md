# Simple Transformers
Simple implementations of transformers to encode and decode most modalities. The transformer, defined in `transformer.py` 
and configured using `config.yaml`, do not change between modalities. The different modalities differ only in their 
processing, which happens in `modality_processors.py`. Some basic example using is shown in `example_usage.py`. 

NOTE: 
the training loops are purely to demonstrate how to use the transformers and should not be used directly.

## Comparison to Hugging Face
[Hugging Face](https://huggingface.co/docs/transformers/index)(HF) is a very well maintained repository containing a huge amount of pre-trained 
models. However, due to HF's focus on pretrained model, it is difficult to use their models for anything that deviates
from the intended usage. For example, their ViLT model requires both image and text as input. If only one of the two is
provided, it throws an error, even though it should be simple to use it to encode a single modality. Further, 
due to it's size, cloning the HF repo to make some small changes requires dealing with an extremely bloated and complex 
repo. This is why I developed simple transformers: it is a lightweight repo that only contains the basics. It covers
enough modalities, so that if there is a modality you require that is not present, it should be straightforward to adapt
one of the existing processors for your needs. In addition, because of its small size, it is easy to clone and adjust
to your needs. This flexibility comes with the downside that pre-trained models are not provided.

## Set Up
Example environment set up:  
1. `conda create -n <env_name>`   
2. `conda activate <env_name>`  
3. install required pytorch setup using conda following: https://pytorch.org/, e.g.  
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`  
5. `pip install -e .`  

## TODOs
- Make quickstart guide from example usages
  - Fix problems in example usage (e.g. val/test sets, zero out decoding loss where appropriate)
  - Turn example usage into a proper script (if __name__ == "__main__" etc) with args. And tell folks to run python scripts/example_usage.py etc.
  - Move this into a Colab that imports the library and runs end to end.
