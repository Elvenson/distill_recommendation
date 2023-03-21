# Distill Recommendation
This repository contains my knowledge and experience in building recommendation systems. This repo is structured in a 
casual way where it works like a notebook. It includes a list of references and some layer/model implementations I have used in my work. For each reference, I will try to link some useful links to help others understand the subject more deeply.

This repo is for everyone. It can be for someone who wants to get into the recommendation field or someone who needs a primer for
their next interviews. I will make this repo as minimal a setup as possible so that others can quickly jump in and
learn from it.

**Note**: This project is ongoing, so the references and code implementation will keep growing.

## Setup
Simply go to this repo directory and type:
```
pip install -r requirements.txt
```

## References
- Transformer layer: [Paper](https://arxiv.org/pdf/1706.03762.pdf), 
[simplified implementation](https://github.com/Elvenson/distill_recommendation/blob/main/layers.py#L82), 
[visual explanation](http://jalammar.github.io/illustrated-transformer/), 
[tensorflow tutorial](https://www.tensorflow.org/text/tutorials/transformer).
- Auto discretize layer: [Paper](https://arxiv.org/pdf/2012.08986.pdf),
[simplified implementation](https://github.com/Elvenson/distill_recommendation/blob/main/layers.py#L111).
- Batch Normalization layer: [Paper](https://arxiv.org/pdf/1502.03167.pdf),
[visual explanation](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739),
[official implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization).
- Layer Normalization layer: [Paper](https://arxiv.org/pdf/1607.06450.pdf),
[official implementation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).