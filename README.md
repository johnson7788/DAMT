# DAMT
神经机器翻译（NMT）的半监督领域适应性的一种新方法

This is the source code for the paper: [Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2020). Unsupervised Domain Adaptation for Neural Machine Translation with Iterative Back Translation. ArXiv, abs/2001.08140.](https://arxiv.org/abs/2001.08140). If you use the code, please cite the paper:

```
@article{Jin2020UnsupervisedDA,
  title={Unsupervised Domain Adaptation for Neural Machine Translation with Iterative Back Translation},
  author={Di Jin and Zhijing Jin and Joey Tianyi Zhou and Peter Szolovits},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.08140}
}
```

## Prerequisites:
运行以下命令以安装先决条件包：
```
pip install -r requirements.txt
```
你还应该通过运行以下命令在 "工具 "文件夹中安装Moses tokenizer和fastBPE工具。
```
cd tools
git clone https://github.com/moses-smt/mosesdecoder
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
cd ../..
```

## Data:
请下载数据从 [Google Drive](https://drive.google.com/file/d/1aQOXfcGpPbQemG4mQQuiy6ZrCRn6WiDj/view?usp=sharing) 
将其解压缩到此repository的主目录 .下载的数据包括DE-EN语言对的MED（EMEA）、IT、LAW（ACQUIS）和TED领域以及EN-RO语言对的MED、LAW和TED领域。 
WMT14 DE-EN data can be downloaded [here](https://nlp.stanford.edu/projects/nmt/) and WMT16 EN-RO data is downloaded from [here](https://www.statmt.org/wmt16/translation-task.html).

下载后解压完成的数据集目录
```buildoutcfg
data
├── de-en
│   ├── acquis
│   │   ├── dev.de
│   │   ├── dev.en
│   │   ├── test.de
│   │   ├── test.en
│   │   ├── train.de
│   │   ├── train.de.mono   monolingual单语数据集, 我们训练时假设只用单语中的一个
│   │   ├── train.en
│   │   └── train.en.mono   monolingual单语数据集
│   ├── emea
│   │   ├── dev.de
│   │   ├── dev.en
│   │   ├── test.de
│   │   ├── test.en
│   │   ├── train.de
│   │   ├── train.de.mono
│   │   ├── train.en
│   │   └── train.en.mono
│   ├── it
│   │   ├── dev.de
│   │   ├── dev.en
│   │   ├── test.de
│   │   ├── test.en
│   │   ├── train.de
│   │   ├── train.de.mono
│   │   ├── train.en
│   │   └── train.en.mono
│   └── ted
│       ├── dev.de
│       ├── dev.en
│       ├── test.de
│       ├── test.en
│       ├── train.de
│       ├── train.de.mono
│       ├── train.en
│       └── train.en.mono
└── en-ro
    ├── acquis
    │   ├── dev.en
    │   ├── dev.ro
    │   ├── test.en
    │   ├── test.ro
    │   ├── train.en
    │   ├── train.en.mono
    │   ├── train.ro
    │   └── train.ro.mono
    ├── emea
    │   ├── dev.en
    │   ├── dev.ro
    │   ├── test.en
    │   ├── test.ro
    │   ├── train.en
    │   ├── train.en.mono
    │   ├── train.ro
    │   └── train.ro.mono
    └── ted
        ├── dev.en
        ├── dev.ro
        ├── test.en
        ├── test.ro
        ├── train.en
        ├── train.en.mono
        ├── train.ro
        └── train.ro.mono
```

## How to use
1. 首先，我们需要下载预训练的模型参数文件，从 [XLM repository](https://github.com/facebookresearch/XLM#pretrained-xlmmlm-models).
下载英语到德语或英语到罗马语
English-German 	MLM 	Model 	BPE codes 	Vocabulary
English-Romanian 	MLM 	Model 	BPE codes 	Vocabulary

#如果要用其它语言翻译模型，更改XLM为XLM-R的模型即可

eg:
wget https://dl.fbaipublicfiles.com/XLM/mlm_ende_1024.pth
wget https://dl.fbaipublicfiles.com/XLM/codes_ende
wget https://dl.fbaipublicfiles.com/XLM/vocab_ende

#英语到德语的预训练模型
```buildoutcfg
model/
├── codes_ende    BPE code
├── mlm_ende_1024.pth   模型
└── vocab_ende  单词表
```

2. 然后我们需要处理这些数据。假设我们要训练从德语（de）到英语（en）的NMT模型，源域是法律（数据集名称是acquis），目标域是IT，那么运行以下命令。
```
bash get-data-nmt-local.sh --src de --tgt en --data_name acquis --data_path ./data/de-en/acquis --reload_codes model/codes_ende --reload_vocab model/vocab_ende
bash get-data-nmt-local.sh --src de --tgt en --data_name it --data_path ./data/de-en/it --reload_codes model/codes_ende --reload_vocab model/vocab_ende
```

3. 在数据处理之后，为了重现论文中提到的 "IBT",迭代的逆向翻译训练， 设置，运行以下命令。
```
bash train_IBT.sh --src de --tgt en --data_name it --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```

4. 为了重现 "IBT+SRC "的设置，回顾一下，我们要从法律域适应IT域，其中源域是法律（数据集名称是acquis），目标域是IT，然后运行以下命令。
```
bash train_IBT_plus_SRC.sh --src de --tgt en --src_data_name acquis --tgt_data_name it --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```

5. 为了重现 "IBT+back "的设置，我们需要经过几步:(直接做这个实验就可以，效果最好)

* 首先，我们需要训练一个NMT模型，通过运行以下命令，使用源域数据（acquis）将en翻译成de， 这里模拟我们有一个最基本的原始域的模型。
```
bash train_sup.sh --src en --tgt de --data_name acquis --pretrained_model_dir model
```
训练后的模型保存到当前目录下的tmp/sup_acquis_en_de文件夹下

* 训练完这个模型后，我们通过使用这个模型将目标域（it）中的英语句子翻译成德语来获得翻译结果，这些句子被作为逆向翻译数据。
```
bash translate_exe.sh --src en --tgt de --data_name it --model_name acquis --model_dir DIR_TO_TRAINED_MODEL
bash get-data-back-translate.sh --src en --tgt de --data_name it --model_name acquis
```

* 当逆向翻译的数据准备好后，我们终于可以运行这个命令了。
```
bash train_IBT_plus_BACK.sh --src de --tgt en --src_data_name acquis --tgt_data_name it --pretrained_model_dir model
```
