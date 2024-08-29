<div align="center">
    
# PANDA-Train-Track

</div>

**PANDA-Train-Track** as part of the PANDA machine learning for tracking project is a heavily modified version of the [train-track repository](https://github.com/murnanedaniel/train-track). It implements a way to execute different stages of the machine learning pipeline via the command line using YAML files for configuration.

## Install

Installation should be done via one of the conda environment files in the [`stttrkx/envs`](../envs) directory. If you want to install an editable stand-alone version execute

```bash
git clone https://github.com/n-idw/panda-train-track.git
```
to download the repository and then

```bash
pip install -e panda-train-track
```

to install PANDA-Train-Track using the [pip package installer](https://pip.pypa.io/en/stable/index.html).

## Objective

The aim of TrainTrack is simple: Given any set of self-contained [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) modules, run them in a serial and trackable way. 

## Example

At its heart, TrainTrack is nothing more than a loop over the stages defined in a YAML configuration file. A template for a YAML file containing the configuration for the pipeline and different stages can be found in [`stttrkx/configs/pipeline_example.yaml`](../configs/pipeline_example.yaml). The model configuration is also done using YAML files in the `/configs` folder of every PyTorch Lightning module. Example configureation files should be present in every of these folders, e.g., [`stttrkx/LightningModules/Processing/configs/processing_example.yaml`](../LightningModules/Processing/configs/processing_example.yaml) for the processing stage.


To launch **traintrack** and see all the implemented options run:

```bash
traintrack -h
```

The simplest way to run a pipeline would be:

```bash
traintrack path/to/your/pipeline_config.yaml
```

## Module Structure

**traintrack** assumes a certain directory & code structure when configuring different . stages. If a stage is configured in the YAML file as follows: 

```yaml
model_library : /path/to/pyTorchLightingModules

stages:
    - {
        set    : stageDir,
        name   : className,
        config : modelConfig.yaml
      }
```

`traintrack` assumes that the directory structure is the following:

```
📂 /path/to/pyTorchLightingModules/
├── 📂 stageDir/
│ ├── 📂 configs/
│ │ ├── 📜 modelConfig.yaml
│ │ └── ...
│ ├── 📂 Models/
│ │ ├── 📜 modelFile1.py
│ │ ├── 📜 modelFile2.py
│ │ ├── 📜 modelFile3.py
│ │ └── ...
└──...
```

And that one of the `modelFiles.py` contains a class with the name `className`. Furthermore it is assumed that the class either has a function `prepare_data()` for processing data, or `training_step()` for training and inference.