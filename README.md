# Biomedical Entity Linking n (Joint4E-EL) Model For [Joint event extraction and entity linking model](https://arxiv.org/abs/2305.14645)
- A deep leanring framework with BERTs and classification layer to link named entities to a knowledge base. The EL model results are reported in our [paper](https://arxiv.org/abs/2305.14645)

## Model Structures
- Based on [Pretrained BERT](https://github.com/allenai/scibert) as encoder.
- Using classification layer to disambiguate candidate concepts from knowledge base.
- Coorparating event structures from event extraction model [(Joint4E-EE)](https://github.com/lxc-dolphin/BioJEE).
- Integrating event-enhanced input/information by inserting the function/role/type of each entity after entity context.
  
<p align="center">
    <br>
    <img src="https://github.com/lxc-dolphin/BioJEL/blob/main/sup/fig_git_EL.png" width="900"/>
    <br>
<p>

## Tasks
- Joint4E-EL model has been trained and evaluated on the following tasks.
1. BioCreative IV GO [((BC4GO)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4112614/)
2. NCBI Disease [(NCBI)](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/)

## Requirements
- Python 3.8
- Pytorch (torch==1.1.0 torchvision==0.3.0, cuda92)
- Install packages

```bash
sh install_EL.sh
```

## Training, Evaluation and Prediction
If using GPU: [-gpu] = 0 and [-cuda] = "True", otherwise: [-gpu] = -1 # default setting: using GPU

data direction: [-data_dir] = ./$$$/###

- training
training without external knowledge or training the baseline or with external knowledge
```bash
python run_e2e_span.py  -data_dir ./$$$/### -do_train -use_SOTA_model True
```

- evaluation
```bash
python run_e2e_span.py  -data_dir ./$$$/### -do_eval -use_SOTA_model True
```

- prediction
predict a single sentence or text file or for event extraciton
```bash
python run_e2e_span.py  -data_dir ./$$$/### -do_test -use_SOTA_model True
```


