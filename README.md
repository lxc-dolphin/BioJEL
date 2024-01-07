# Biomedical Entity Linking n (Joint4E-EL) Model For [Joint event extraction and entity linking model](https://arxiv.org/abs/2305.14645)
- A deep leanring framework with BERTs and classification lay to link named entities to a knowladge base. The EL model results are reported in our [paper](https://arxiv.org/abs/2305.14645)

## Model Structures
- Based on [Pretrained BERT](https://github.com/allenai/scibert) as encoder.
- Using classification layer to .
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
If using GPU: [-gpu] = 0 and [-cuda] = "True", otherwise: [-gpu] = -1

- training
1. training without external knowledge or training the baseline
```bash
python run_e2e_span.py -do_train True -add_candi False -gpu 0 -cuda True
```
2. training with external knowledge 
```bash
python run_e2e_span.py -do_train True -add_candi True -gpu 0 -cuda True -use_SOTA_model True
```


- evaluation
```bash
python run_e2e_span.py -do_eval True -add_candi False -gpu 0 -cuda True -use_SOTA_model True
```

- prediction
1. predict a single sentence or text file
```bash
python run_e2e_span.py -do_test_sihgle True -use_SOTA_model True
```
or 
```bash
python run_e2e_span.py -single_test_file -test_input_file [file name] -use_SOTA_model True
```

2. predict event for entity linking task
```bash
python run_e2e_span.py -do_test_ELdata True -use_SOTA_model True
```
