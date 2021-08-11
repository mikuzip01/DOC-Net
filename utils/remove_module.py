from collections import OrderedDict


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict