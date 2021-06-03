# MultiMedia and Machine Learning

A framework for building, experimenting with, and deploying multimedia restoration/enhancement algorithms based on machine learning.

This is pre-alpha code with no version numbers. The following sparse documentation is intended for a small set of developers to help build out the library. Code is currently private and available only by invitation. 

## Overview

At a high level the project is using python with pytorch powering the machine-learning side.

The number one goal of this project is to minimize the amoint of boilerplate and systems-level programming both inside the library itself and for end-users. It is often tricky to solve these engineering problems and there are high quality purpose built libraries for almost everything in python these days. This also makes the code smaller and therefore easily maintainable. If we develop something we need for the project and it seems like it isn't directly related, it should either be spun off into its own library or contributed to another project (see torchjpeg for an example of that).

To that end, we are *prescribing* the following libraries (i.e., they are mandatory) 

* Pytorch for deep learning
* Pytorch-lightning to remove deep learning boilerplate
* Hydra for complex configuration and command line arguments (including distributed launches and hyperparameter sweeps)

We are also *recommending* the following libraries 

* slurm for cluster computing (can be replaced by any launcher supported by hydra)
* optuna for hyperparameter sweeps (can be replaced by any sweeper support by hydra)
* WandB for experiment tracking and lifecycle management (can be replaced by any logger pytorch-lightning supports)
* torchhub for pretrained weights

By recommending we mean that code that is part of the project will be designed to use these librarie. Although it should not crash if presented with an alternative, it may not work completely. End-user model/dataset code that uses alternative libraries with M3L used to do the training/evaluations should work as expected. 

## Long Term Goals

* Incorporate a full set of restoration models (denoising, deblurring, compression correction, etc) for both images and videos (this will likely **not** include super resolution, BasicSR is already good for that)
* Optional torchscript models for production deployments with minimal dependencies
* Hyperparameter sweeps with optuna
* torchhub for pre-trained weights

## Optimizing Your Own Model

Define your model in a LightningModule (see lightning docs)

Define your dataset in a LightningDataModule (see lightning docs) or use a built-in dataset. M3L uses `predict_dataloader` for correction, so make sure to define that along with a parameter like `correct_dir` that allows you to pass an input directory. If you just want to do trainig/evaluations then you don't need this.

Configure m3l, it uses `~/.config/m3l` or `/etc/m3l`:

The network config (`<config_dir>/net`) tells m3l how to instantiate your network, here's a template:

```
_target_: my_library.my_package.MyRestorationNetwork
n_layers: 10
width: 50
epsilon: 0.5
...
```

Asssuming that your model takes `n_layers`, `width`, etc as parameters to `__init__`. See hydra docs for more info on how this works and how to override these on the command line.

If you have a custom dataset, make a config for that (`<config_dir>/data`) like this:

```
_target_: my_library.my_package.MyDataset
root_dir: /data/someplace
train_batch_size: 32
num_workers: 1
```

Finally, make a model config (`<config_dir>/model`), this tells M3L about what it needs to do to train and evaluate your model:

```
# @package _global_
defaults:
  - /data: my_dataset  # whatever you named the yaml file for the dataset (or one of the built-ins)
  - /net: my_net  # whatever you named the yaml file for the network
  - override /serializer: image_serializer  # Required for correcting images after training/evaluation

trainer:
  max_epochs: 200  # Override any parameters for the pytorch lightning `trainer` here

name: my_network  # name of your model (used for logging)
```

Note that `# @package _global_` on the first line is mandatory. You can override pretty much anything that M3L does with the above model defintion. For example you can override properties of the network, the trainer (see pytorch lightning) the dataset, or even hydra internals. 

Next, train your model. Make sure `PYTHONPATH` is set correctly so that M3L can find your code before doing this. 

To run training locally:

`python -m m3l.run.train model=my_model`

To run testing locally

`python -m m3l.run.test model=my_model checkpoint=<path to checkpoint>`

To correct images locally

`python -m m3l.run.correct model=my_model checkpoint=<path to checkpoint> data.correct_dir=<path to directory of images or single image> serializer.root_dir=<path to output root>`

## Logging

By default, M3L will not set up any loggers. This causes pytorch-lightning to save logged tensors using tensorboard summaries in the pwd. If you want to use WandB, override the `logger` package to be `wandb` and log in following wandb docs.

Example:

`python -m m3l.run.train model=my_model logger=wandb`

This can also be saved to a config file (for example, in your model definition):

```
# @package _global_
defaults:
  ...
  - /logger: wandb
```

If you want to use another logger, you need to set up the configs for it yourself.

## Compute Cluster

M3L can automatically launch your job on a compute cluster. This works for training, testing, and correction. **Note** that distributed testing requires great care because of how pytorch's distributed sampler works. Unless you really know what your are doing use only a single GPU for testing, otherwise your metrics may be incorrect.

Define a compute cluster using a config file in `<config dir>/cluster`, here is a simple example that will work on the vulcan cluster at UMD:

```
# @package _global_
defaults:
 - override /hydra/launcher: submitit_slurm

hydra:
  launcher:
    name: ${name} 
    timeout_min: 2160 # 1.5 days
    tasks_per_node:  4
    nodes: 1  
    mem_gb: 128
    cpus_per_task: 4
    additional_parameters:
      qos: high
      gres: gpu:gtx1080ti:4
```

This config file uses the `submitit_slurm` hydra launcher to manage your job on the cluster. This will max out the high QOS that vulcan provides for every job. You will also need to set `trainer.gpus` accordingly (command line or config).

Hydra supports a very powerful interface for configuration and M3L allows you to save arbitrary cluster data in the `cluster` key to refer to later, we can use this to make a smarter cluster config:

```
# @package _global_
defaults:
 - override /hydra/launcher: submitit_slurm

cluster:
  qos: high
  gpu_type: p6000
  # mins per QOS
  qos_timeout:
    high: 2160
    medium: 4320
    default: 10080

hydra:
  launcher:
    name: ${name} 
    timeout_min: ${cluster.qos_timeout.${cluster.qos}}
    tasks_per_node:  ${oc.select:trainer.gpus,1}
    nodes: ${oc.select:trainer.nodes,1}
    mem_gb: ${m3l.mul:${.tasks_per_node},32}
    cpus_per_task: 4
    additional_parameters:
      qos: ${cluster.qos}
      gres: gpu:${cluster.gpu_type}:${..tasks_per_node}
```

This config automatically scales the number of GPUs and memory based on the number of GPUs you pass to the trainer and allows you to select different QOS and GPU types (cluster.qos and cluster.gpu_type respectively). 

Assuming this file is saved as `<config dir>/cluster/vulcan.yaml`, launch distributed training (on the **submission node** as follows)

`python -m m3l.run.train model=my_model cluster=vulcan trainer.gpus=4`

This will request 4 GPUs and submit your job to slurm, it will tell you the job's directory where you can find stdout/stderr models, etc. If you're using wandb you can also view stdout/stderr on the dashboard there.

## Adding a Model/Dataset to M3L

Adding to M3L is pretty much the same process as the Own Code section except you put your model in the repository `models/`, your dataset at `data/`, and your configs in `configs/`. Aside from that, verify that the model works with WandB and slurm (and anything other recommended libraries). For now there's no process around pre-trained weights. 