# (Simplified) Solution to Recruit Competition (2018)


Sorry, no CPU-only mode. You have to use an nVidia card to train models.

Test environment:
1. GTX 1070
2. 16 GB RAM + 8 GB Swap
3. At least 2 GB free disk space
  - (it can be less if you turn off some of the joblib disk caching)
4. Docker 17.12.0-ce
5. Nvidia-docker 2.0


This is based on my solution to [the Favorita competition](https://github.com/ceshine/favorita_sales_forecasting). Some of the modifications:

1. Removes the weighted loss calculation based on the *perishable* property.
2. Create a masked MSE criterion to ignore missing entries in target sequences.
3. Add support for [NT-ASGD](https://arxiv.org/abs/1708.02182).


## Acknowledgement

1. Transformer model comes from [Yu-Hsiang Huang's implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch). His repo is included in "*attention-is-all-you-need-pytorch*" folder via *git subtree*.
2. The model structure is inspired by the work of  [Sean Vasquez](https://github.com/sjvasquez/web-traffic-forecasting) and [Arthur Suilin](https://github.com/Arturus/kaggle-web-traffic).
3. NT-ASGD algorithm is introduced in the [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) paper and its official [implementation](https://github.com/salesforce/awd-lstm-lm).
4. [This script by Matthew Dangerfield](https://gist.github.com/superMDguy/72689a11746079677ddb0d19f26443a1) helped me a lot in writing code to import weather data.

## Docker Usage

First build the image. Example command: `docker build -t recruit .`

Then spin up a docker container:
```
docker run --runtime=nvidia --rm -ti \
    -v /mnt/Data/recruit_cache:/home/docker/labs/cache \
    -v /mnt/Data/recruit_data:/home/docker/labs/data \
    -p 6006:6006 recruit bash
```

* It is recommended to manually mount the data and cache folder
* port 6006 is for running tensorboard inside the container

### Where to put the data
Download and extract the [offical data files from Kaggle](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data) and the [external weather data](https://www.kaggle.com/huntermcgushion/rrv-weather-data) into `data` folder.

We're going to assume you're using the BASH prompt inside the container in the rest of this README.

## Model Training

### Preprocessing

```
python preprocess.py
```

### Train Model

For now there are only one type of model ready to be trained:
1. Transformer (fit_transformer.py)

The training scripts use [Sacred](http://sacred.readthedocs.io/en/latest/) to manage experiments. It is recommended to set a seed explicitly via CLI:

```
python fit_transformer.py with seed=12300
```

(To give you an rough idea of how a single model performs, the above command created a model with **Public 0.488 / Private 0.517**. Sorry for now it's not exactly reproducible.)

You can also use Mongo to save experiment results and hyper-parameters for each run. Please refer to the Sacred documentation for more details.

### Prediction for Validation and Testing Dataset

The CSV output will be saved in `cache/preds/val/` and `cache/preds/test/` respectively.

### Tensorboard

Training and validation loss curves, and some of the embeddings are logged in tensorboard format. Launch tensorboad via:

```
tensorboard --logdir runs
```

Then visit http://localhost:6006 for the web interface.

## TODO (For now you need to figure them out yourself)

1. Ensembling script: I made some changes to the outputs of model training scripts so they are more readable. But that means ensembling script needs to be updated as well. (For those who want to try: the ground truth for validation set is stored in `cache/yval_seq.npy`.)
2. Reproducibility: I somehow broke it when porting the code from my private repo to this public one. The models were reproducible with the same seed.
3. Weaker models: Both public and private score can be boosted by adding more weaker models to the mix, e.g. Prophet, LightGBM, Weight Means.
