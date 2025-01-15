# Delphi - Secure Application of Distributed Learning in Metaverse: Creating tools for Trustworthiness of AI

Delphi employs multiple optimisation techniques - Bayesian Optimisation (BO), Multi-Objective Bayesian Optimisation (MOBO), Least Squares Trust Region (LSTR), and Reinforcement Learning (RL) - to search for optimal poisoned model parameters that maximise uncertainty in the learning system. The framework can target specific neurons and layers to conduct stealthy attacks that are difficult to detect. Through extensive experiments on various datasets and model architectures, including VGG11, ResNet18, MobileNet and Vision Transformers, Delphi demonstrates significant capability to degrade model performance and increase uncertainty whilst maintaining attack stealth.

## Features

- Model Poisoning Attack using 4 different technique to create poisonous parameters
- Graph analysis for understanding the relationships between the clients

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/GrMrWb/Delphi.git
cd Delphi
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
source venv/bin/activate
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

## Command Line Arguments 
This arguments are settings which you can manipulate on each FL

#### General FL Settings

#### Dataset Settings
| Short Flag | Long Flag | Values/Type | Default | Description |
| :----------: |-----------|-------------|-------------|-------------|
| `-d` | `--dataset` | CIFAR10 / CIFAR100 / SVHN / DomainNet / ISIC2019 | CIFAR10 | The dataset in use |
| `-m` | `--model` | Many DL (specified below) | AlexNet | The model in use |
| `-p` | `--iid` | 1=IID, 0=non-IID | 0 | IID or non-IID data distribution |
| `-tp` | `--partition` | dirichlet / pathological / imbalanced | imbalanced | Partition type |
| `-da` | `--dirichlet_alpha` | float (0 to 1) | 0.1 | Dirichlet alpha parameter |
| `-ia` | `--imbalanced_ratio` | float (0 to 1) | 0.25| Imbalanced ratio |
| `-bstr` | `--training_batch_size` | integer | 128 | Training batch size |
| `-bsts` | `--testing_batch_size` | integer | 256 | Testing batch size |
| `-vs` | `--valid_size` | float | 0.15| Validation Size |

#### Attack Settings
| Short Flag | Long Flag | Values/Type | Default | Description |
| :----------: |-----------|-------------|-------------|-------------|
| `-aa`      | `--active_attack`        | 1=attack, 0=no attack | 1 | Enable/disable attack, if there is no adversarial clients wont engage but the filename will change |
| `-ec`      | `--engagement_criteria`  | integer | 1 | Epoch to engage |
| `-ta`      | `--type_of_attack`       | single / multi | single |Attack type |
| `-ad`      | `--available_data`       | min / max / random | random | Data availability type |
| `-to`      | `--type_of_optimisation` | least_squares / bayesian_optimisation / rl | bayesian_optimisation |Optimization method |
| `-o`       | `--objective`            | kl_div / mutual_info / js_div | kl_div | Objective function |
| `-obj1`    | `--objective_1st`        | kl_div / mutual_info / js_div | kl_div | First objective function |
| `-obj2`    | `--objective_2nd`        | kl_div / mutual_info / js_div / accuracy | mutual_info |Second objective function |
| `-n`       | `--neurons`              | integer | 5 | Number of Neurons |
| `-fi`      | `--features_indices`     | 0=Continues, 1=Fixed | 0 | Indices of Neurons |
| `-ul`      | `--use_logits`           | 0=False, 1=True | 0 | Use logits flag |
| `-ab`      | `--attack_bounds`        | float | 1 | Attack bounds |
| `-al`      | `--attack_layers`        | format: 1.2.3 | 1.2 | Layers to attack seperated by a dot |

#### General settings for the Experiment
| Short Flag | Long Flag | Values/Type | Default | Description |
| :----------: |-----------|-------------|-------------|-------------|
| `-ns` | `--number_of_seeds` | integer | 1 | Number of seeds |
| `-de` | `--device` | cpu/cuda | cuda | Computing device (if GPU is available) |
| `-ch` | `--checkpoint` | 0=False, 1=True | 0 | Start from checkpoint |
| `-doe` | `--date_of_experiment` | YYYYMMDD format | 00000000 | Date of experiment (use the exact date when the checkpoint is enabled) |

### Example Usage

Basic usage:
```bash
python main.py -d CIFAR10 -m AlexNet -nc 10 -tr 100 -p 1
```

With attack configuration:
```bash
python main.py -d CIFAR10 -m AlexNet -nc 10 -na 2 -tr 100 -p 0 -tp dirichlet -da 0.5 -aa 1 -ta single
```

Full configuration with optimization and attack layers:
```bash
python main.py -d CIFAR10 -m AlexNet -nc 10 -na 2 -tr 100 -le 5 -p 0 \
               -tp dirichlet -da 0.5 -bstr 64 -bsts 128 -aa 1 \
               -ta single -to bayesian_optimisation -o kl_div \
               -al 1.2.3 -de cuda
```

### Models available
We have multiple models which we have used and tested. Thesre are the following:
 - AlexNet
 - MLP_MNIST, MLP_CIFAR10, MLP_CIFAR100
 - Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
 - SwinTransformer, VisionTransformer
 - VGG1 , VGG13, VGG16, VGG19
 - MobileNetv3


## Project Structure

```
## Project Structure

Delphi
├── config
│   └── config_server.yaml
├── src
│   ├── dataset
│   │   ├── data.py
│   │   ├── source
│   │   └── utils
│   ├── delphi
│   │   ├── convex_opt
│   │   ├── rl
│   │   ├── strategy.py
│   │   ├── ubo
│   │   └── utils
│   ├── evaluation
│   └── learning
│       ├── client
│       ├── models
│       ├── server
│       └── utils
├── application_logs
├── README.md
├── config.py
├── main.py
├── requirements.txt
└── scripts.sh
```

<!-- ## License

This project is licensed under the [LICENSE NAME] - see the [LICENSE](LICENSE) file for details. -->

<!-- ## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/username/project-name](https://github.com/username/project-name) -->

## Use of the code
In case you have used this code, please give us some credit and cite the following paper:
```
@Article{Aristodemou2024TIFS,
  author  = Aristodemou, Marios and Liu, Xiaolan and Wang, Yuan and Konstantinos G., Kyriakopoulos and Lambotharan, Sangarapillai and Wei, Qingsong},
  journal = {IEEE Transactions on Informations Forensics and Security},
  title   = {Maximising Uncertainty for Federated Learning via Baeysian Optimisation-based Model Poisoning},
  year    = {Forthcoming},
}
```

## Acknowledgments
- pFLLIB has been used in order to make use of the FL algorithms.
- Special thanks to contributors
- This code has been used for my PhD Studies
