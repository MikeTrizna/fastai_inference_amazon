# Amazon fish inference

## Download fish photos and fish model

```
wget -O Peru_Fish_ID.pkl https://www.dropbox.com/s/kh91iu9yruwbfjv/Peru_Fish_ID.pkl?dl=1
wget -O fish_november.zip https://www.dropbox.com/sh/p3bbk3rlt4xvo60/AAD43s6VotvOaIFTkqXQcv36a?dl=1
unzip -q fish_november.zip
```

## Create conda environment

```
conda env create -f environment.yml
conda activate fastai_cpu
```

## Classify fish photos

```
python fish_classifier.py -d amazon_fish -o november_2019_results.tsv
```

## Analyze results

Launch a Jupyter notebook and run through `calculating results.ipynb`

`jupyter notebook`