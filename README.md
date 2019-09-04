# knowledge-distillation
## description
in this repo, I have written and refactored some code to compact resnet model using knowledge distillation
## references
https://arxiv.org/abs/1503.02531  
https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322
https://towardsdatascience.com/turning-up-the-heat-the-mechanics-of-model-distillation-25ca337b5c7c
## requirement
CUDA 9.0 + Pytorch 1.0
## run code
ROOT = path/to/this/repo
- download dataset
```
cd ROOT
mkdir dataset
```
link download: https://download.pytorch.org/tutorial/hymenoptera_data.zip
- train and evaluate
```python
cd ROOT
python main.py
