# YouyakuMan

 [![Unstable](https://poser.pugx.org/ali-irawan/xtra/v/unstable.svg)](*https://poser.pugx.org/ali-irawan/xtra/v/unstable.svg*)  [![License](https://poser.pugx.org/ali-irawan/xtra/license.svg)](*https://poser.pugx.org/ali-irawan/xtra/license.svg*)

### Introduction

This is an one-touch extractive summarization machine.

using BertSum as summatization model, extract top N important sentences.

![img](https://cdn-images-1.medium.com/max/800/1*NRamBWCtYuS8U6pqpnDiJQ.png)

---

### Prerequisites

#### General requirement

```
pip install torch
pip install transformers
pip install googletrans
```

#### Japanese specific requirement

- [BERT日本語Pretrainedモデル — KUROHASHI-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル)
- [Juman++ V2の開発版](https://github.com/ku-nlp/jumanpp)[ — KUROHASHI-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル)

---

### Pretrained Model

English: [Here](https://drive.google.com/open?id=1wxf6zTTrhYGmUTVHVMxGpl_GLaZAC1ye)

Japanese: [Here](https://drive.google.com/file/d/1BBjg0LI8VAgpKT6QN1ah1S49mlUhbM1h/view?usp=sharing)

* Japanese model updated: trained with 35k data and 120k iteration

Download and put under directory `checkpoint/en` or `checkpoint/jp`

---

### Example

```
$python youyakuman.py -txt_file YOUR_FILE -lang LANG -n 3 --super_long
```

#### Note

Since Bert only takes 512 length as inputs, this summarizer crop articles >512 length.

If --super_long option is used, summarizer automatically parse to numbers of 512 length inputs and summarize per inputs. Number of extraction might slightly altered with --super_long used.

---

### Train Example

```
$python youyakumanJPN_train.py -data_folder [training_txt_path] -save_path [model_saving_path] -train_from [pretrained_model_file]
"""
-data_folder : path to train data folder, structure showed as below:
                training_txt_path
                ├─ article1.pickle
                ├─ article2.pickle
                ..    
"""
```

### Train Data Preparation

Training data should be a dictionary saved by `pickle`, to be specifically, a dictionary containing below contents of **one article**.

```
{'body': 'TEXT_BODY', 'summary': 'SUMMARY_1<sep>SUMMARY_2<sep>SUMMARY3'}
```

---
### Version Log:

2020-08-03  Updated to `transformer` package, remove redudndancy, model saving format while training

2020-02-10  Training part added

2019-11-14  Add multiple language support

2019-10-29 	Add auto parse function, available for long article as input
