## Automated Utterance Labeling of Conversations Using Natural Language Processing

This repository contains the code for EMO-COG and EMO8 classification using our pretrained model, RoBERTa-CON. 

## RoBERTa-CON 
Our model was pretrained on [Alexander Street Press dataset of counseling and psychotherapy transcripts](https://alexanderstreet.com/products/counseling-and-psychotherapy-transcripts-series). More information about the training process, parameters used and performance is avaliable in our [paper](https://arxiv.org/pdf/2208.06525.pdf). RoBERTa-CON can be uploaded from [Transformers Hub](https://huggingface.co/mlaricheva/roberta-psych).  

You can load this model by:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("mlaricheva/roberta-psych")
model = AutoModelForMaskedLM.from_pretrained("mlaricheva/roberta-psych")
```

## Citation
More details can be found in our [paper](https://link.springer.com/chapter/10.1007/978-3-031-17114-7_23):
```
@InProceedings{10.1007/978-3-031-17114-7_23,
author="Laricheva, Maria and Zhang, Chiyu and Liu, Yan and Chen, Guanyu and Tracey, Terence and Young, Richard and Carenini, Giuseppe",
title="Automated Utterance Labeling of Conversations Using Natural Language Processing",
booktitle="Social, Cultural, and Behavioral Modeling",
year="2022",
publisher="Springer International Publishing",
isbn="978-3-031-17114-7"
}
```
