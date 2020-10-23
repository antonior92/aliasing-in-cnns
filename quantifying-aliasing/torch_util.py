import torch
from collections.abc import Iterable


def _get_layers(model, all_layers=None, all_names=None, top_name=None, fn=None, sep='_'):
    """Auxiliar function. Recursive method for getting all in the model for which `fn(layer)=True`."""
    if all_names is None:
        all_names = []
    if all_layers is None:
        all_layers = []
    if top_name is None:
        top_name = ''
    if fn is None:
        fn = lambda l: True
    for name, layer in model.named_children():
        if list(layer.children()):
            all_names, all_layers = _get_layers(layer, all_layers, all_names, top_name+name+sep, fn)
        else:
            if fn(layer):
                all_names.append(top_name + name)
                all_layers.append(layer)
    return all_names, all_layers


def get_layers(model, fn=None, sep='_'):
    """Get all layers of torch.nn.Module for which `fn(layer)=True` using a depth-first search.
    Given the module `model` and the function `fn(layer: torch.nn.Module) -> bool` return all layers
    for which the function returns true. Return a list of tuples: ('name', Module). For nested blocks
    the name is a single string, with subblocks names separed by `sep` (by default `sep=_`). For instance,
    `layer1_0_conv1` for 3 nested blocks `layer1`, `0`, `conv1`."""
    all_names, all_layers = _get_layers(model, fn=fn, sep=sep)
    return list(zip(all_names, all_layers))


def replace_layer(model, layer_name, replace_fn):
    """Replace single layer in a (possibly nested) torch.nn.Module using `replace_fn`.
    Given a module `model` and a layer specified by `layer_name` replace the layer using
    `new_layer = replace_fn(old_layer)`. Here `layer_name` is a list of strings, each string
    indexing a level of the nested model."""
    if layer_name:
        nm = layer_name.pop()
        model._modules[nm] = replace_layer(model._modules[nm], layer_name, replace_fn)
    else:
        model = replace_fn(model)
    return model


def replace_all_layers(model, layers, replace_fn, sep='_'):
    """Replace layers in a (possibly nested) torch.nn.Module using `replace_fn`.
    Given a module `model` and a layer specified by `layer_name` replace the layer using
    `new_layer = replace_fn(old_layer)`. Here `layer_name` is a list of strings, each string
    indexing a level of the nested model."""
    for l in layers:
        model = replace_layer(model, l.split(sep)[::-1], replace_fn)
    return model


class SaveIntermediaryValues(object):
    """Module for saving intermediary values."""

    def __init__(self,  collapsing_fn, is_layer_fn,  n_samples):
        self.collapsing_fn = collapsing_fn
        self.is_layer_fn = is_layer_fn
        self.batch_dim = 0
        self.n_samples = n_samples
        self.counter = None
        self.is_first_execution = None
        self.storage = None
        self.layer_names = None

    def save_forward_hooks(self, model):
        all_layers = get_layers(model, fn=self.is_layer_fn)
        self.layer_names = list(list(zip(*all_layers))[0])
        self.storage = {name: None for name in self.layer_names}
        self.counter = {name: 0 for name in self.layer_names}
        self.is_first_execution = {name: True for name in self.layer_names}
        for name in self.layer_names:
            model = replace_all_layers(model, [name], replace_fn=self.hook(name))
        return model

    def hook(self, name):
        def register_forward_hook(layer):
            def forward_hook(_self, inp, _out):
                x = self.collapsing_fn(inp[0], _self)
                if self.is_first_execution[name]:
                    self.is_first_execution[name] = False
                    self.storage[name] = self.init_storage(x)
                delta = self.update_storage(x, self.storage[name], self.counter[name])
                self.counter[name] += delta
            layer.register_forward_hook(forward_hook)
            return layer
        return register_forward_hook

    def init_storage(self, x):
        if type(x) == torch.Tensor:
            shape = list(x.shape)
            shape[self.batch_dim] = self.n_samples
            return torch.zeros(shape, dtype=x.dtype)
        elif type(x) == dict:
            aux = {}
            for key, value in x.items():
                aux[key] = self.init_storage(value)
            return aux
        elif isinstance(x, Iterable):
            aux = []
            for xx in x:
                aux.append(self.init_storage(xx))
            return tuple(aux)
        else:
            raise NotImplementedError()

    def update_storage(self, x, storage, counter):
        if type(x) == torch.Tensor:
            delta = x.shape[self.batch_dim]
            storage[counter:counter + delta, ...] = x
            return delta
        elif type(x) == dict:
            delta = 0
            for key, value in x.items():
                delta = self.update_storage(value, storage[key], counter)
            return delta
        elif isinstance(x, Iterable):
            delta = 0
            iter_storage = iter(storage)
            for xx in x:
                delta = self.update_storage(xx, next(iter_storage), counter)
            return delta
        else:
            raise NotImplementedError()

    def reset_storage(self, storage=None):
        if storage is None:
            storage = self.storage
        if type(storage) == torch.Tensor:
            storage[...] = 0
        elif type(storage) == dict:
            for key, value in storage.items():
                self.reset_storage(storage[key])
        elif isinstance(storage, Iterable):
            iter_storage = iter(storage)
            for xx in x:
                self.reset_storage(next(iter_storage))
        else:
            raise NotImplementedError()

    def reset(self):
        self.counter = {name: 0 for name in self.layer_names}
        self.is_first_execution = {name: True for name in self.layer_names}
        self.reset_storage()