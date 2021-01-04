## Tool: Hydra 

> The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line

The main objective use Hydra is to speed up our training pipeline by configuration.


Here, will show how to use `Hydra` in your ML/DL project to set up our model hyperparameters.

If you want to know more about how to use `Hydra`, the document is available [here](https://hydra.cc/docs/intro).

### Installation
First, install `hydra-core`.
```
pip install hydra-core --upgrade
```

### Model
In `app.py`, there is an example to build a simple model. Change it to your own model/code at will.

### Config
In `config.yaml` define model hyperparameters. 
For example, in the below, I define the `input_size`,  `output_size`, 
whether I need to `save` the model and other hyperparameters. 

```yaml
model:
  input_size: 1
  output_size: 1
  batch_size: 32
  learning_rate: 0.0003
  epochs: 30
  plot: no
  save:
    need: yes
    version: 'v1'
    name: 'simple_model_${model.save.version}.pkl'
```

Here is an easy way to update config values via the command line. 
* For example, `plot=no` by default, you can change the parameters directly like this：
```
python app.py model.plot=yes
```

