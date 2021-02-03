# Split-and-Bridge
## Split-and-Bridge: Adaptable Class Incremental Learning within a Single Neural Network
Split-and-Bridge: Adaptable Class Incremental Learning within a Single Neural Network 
in AAAI2021 by Jong-Yeong Kim and Dong-Wan Choi

<img src="https://user-images.githubusercontent.com/74110603/98468351-6d6a2180-221d-11eb-96ce-ad416da90100.png" width="85%" height="80%">

## Results
### Average accuracies over all the incrmental tasks of ResNet-18 using CIFAR-100

The following results can be reproduced with command:
    
    python main.py --dataset CIFAR100 --trainer split -- base-classes 50 --step-size 50 --rho 1
    python main.py --dataset CIFAR100 --trainer split -- base-classes 20 --step-size 20 --rho 1.35
    python main.py --dataset CIFAR100 --trainer split -- base-classes 10 --step-size 10 --rho 1.15
    python main.py --dataset CIFAR100 --trainer split -- base-classes 5 --step-size 5 --rho 1


|Number of tasks|2|5|10|20|
|-----------------|---|---|---|---|
|STD with iCaRL|68.02|63.0|58.05|60.36|
|STD with Bic|**70.13**|68.22|61.10|48.68|
|STD with WA|69.72|68.73|63.98|54.93|
|DD with WA|69.34|68.53|63.77|57.03|
|S&B with WA (ours)|69.77|**69.76**|**67.56**|**61.52**|




### Average accuracies over all the incrmental tasks of ResNet-18 using Tiny-ImageNet

The following results can be reproduced with command:
    
    python main.py --dataset TinyImagenet --trainer split -- base-classes 100 --step-size 100 --rho 1
    python main.py --dataset TinyImagenet --trainer split -- base-classes 40 --step-size 40 --rho 1.35
    python main.py --dataset TinyImagenet --trainer split -- base-classes 20 --step-size 20 --rho 1.2
    python main.py --dataset TinyImagenet --trainer split -- base-classes 10 --step-size 10 --rho 1.113
    

|Number of tasks|2|5|10|20|
|-----------------|---|---|---|---|
|STD with iCaRL|55.35|52.05|48.61|46.41|
|STD with Bic|57.90|56.15|49.17|42.43|
|STD with WA|57.49|56.45|52.34|47.10|
|DD with WA|58.21|57.53|53.51|48.15|
|S&B with WA (ours)|**59.0**|**57.64**|**55.11**|**51.64**|




### Comparative performance of Split(1phase), Split+Bridge(1phase + 2phase) and S&B with WA on CIFAR-100

|Number of tasks|1|2|
|-----------------|---|---|
|Split|81.16|62.02|
|Split+Bridge|81.16|66.40|
|S&B with WA|81.16|69.60|

|Number of tasks|1|2|3|4|5|
|-----------------|---|---|---|---|---|
|Split|85.05|75.55|60.11|60.40|54.28|
|Split+Bridge|85.05|77.22|66.38|60.40|54.28|
|S&B with WA|85.05|77.47|69.91|66.29|60.80|

|Number of tasks|1|2|3|4|5|6|7|8|9|10|
|-----------------|---|---|---|---|---|---|---|---|---|---|
|Split|82.60|76.25|71.43|69.08|60.08|50.56|37.89|51.71|49.11|44.64|
|Split+Bridge|82.60|80.05|74.67|72.12|65.86|60.28|57.30|51.71|49.11|44.64|
|S&B with WA|82.60|80.55|76.70|74.97|70.42|65.85|62.70|60.10|57.36|54.08|

|Number of tasks|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
|-----------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Split|78.60|68.0|69.53|71.25|67.16|61.43|63.85|59.52|58.93|47.86|45.56|46.87|44.80|39.61|36.33|35.35|34.25|31.04|30.28|27.09|
|Split+Bridge|78.60|79.40|79.40|76.80|72.04|68.40|68.37|66.75|65.10|55.54|56.05|54.70|54.13|50.08|46.58|45.55|44.38|42.43|39.68|38.27|
|S&B with WA|78.60|78.60|79.87|77.30|75.04|71.07|70.57|69.70|67.97|60.58|60.31|58.63|56.57|54.26|51.09|49.0|47.64|46.82|44.43|41.91|
       
## Usage
### Prerequisites
1. Pytorch
2. Python packages: numpy
### Command
    python main.py --dataset <choose dataset> --trainer <choose trainer>
*Example*: `python main.py --dataset CIFAR100 --trainer split`

### Arguments
*Required*:
* `--dataset`: Choose datset. *Option*: `CIFAR100` or `TinyImagenet`
* `--trainer`:  Choose trainer. *Option*: `split` or `icarl` or `bic` or `wa` or `dd`

*Optional*: 
* `--batch-size`: input batch size for training. *type*: `int`, *Default*: `256`
* `--workers`: Number of workers in Dataloaders. *type*: `int`, *Default*: `0`
* `--nepochs`: Number of epochs for each increment. *type*: `int`, *Default*: `200`
* `--lr`: learning rate. *type*: `float`, *Default*: `0.1`
* `--schedule`: Decrease learning rate at these epochs. *type*: `int`, *Default*: `[60,120,160]`
* `--gammas`: LR is multiplied by gamma on schedule, number of gammas should be equal to schedule. *type*: `float`, *Default*: `[0.1, 0.1,0.1]`
* `--momentum`: SGD momentum. *type*: `float`, *Default*: `0.9`
* `--decay`: Weight decay (L2 penalty). *type*: `float`, *Default*: `0.0005`
* `--base-classes`: Number of base classe. *type*: `int`, *Default*: `20`
* `--step-size`: How many classes to add in each increment. *type*: `int`, *Default*: `20`
* `--memory-budget`: How many images can we store at max. *type*: `int`, *Default*: `2000`
* `--rho`: adaptive split hyperparameter. *type*: `float`, *Default*: `1`
* `--seed`: Seeds values to be used; seed introduces randomness by changing order of classes. *type*: `int`, *Default*: `0`


## Acknowledgements

This implementation has been tested with Pytorch 1.2.0 on Windows 10.

