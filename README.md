# neural_dependence_fields

Requirements:

pytorch-cuda=11.7

tensorboard=2.15.0

tinycudann=1.7

xarray=2023.1.0

Randomly sampling reference and query positions and computing groundtruths:
```
conda activate env
python mi_pearson_sampler.py --source_path path/to/folder/of/.nc/files
```
Run `TrainingScript.py`:
```
python TrainingScript.py
```
