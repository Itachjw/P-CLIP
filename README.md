
# **P-CLIP: Progressive Discrepancy Learning for One-Shot Text-to-Image Person Re-identification** 

## Highlights

Unlike previous work, we focus on a challenging and practically important semi-supervised task called **One-Shot Text-to-Image Person Re-identification** (one-shot TIReID), a potential solution that reduces labeling efforts by using only one labeled image-text pair per identity along with a pool of unlabeled person images.

![](image/one-shot.png)

## Usage

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training

```python
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'ccm+cdl' \
--num_epoch 60
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```
