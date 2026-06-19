import importlib.util
import os
import sys


def get_custom_reward_fn(config):
    reward_fn_config = config.get("custom_reward_function")
    if not reward_fn_config and config.get("reward") is not None:
        reward_fn_config = config.reward.get("custom_reward_function")
    reward_fn_config = reward_fn_config or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Error loading module from '{file_path}': {exc}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    raw_fn = getattr(module, function_name)
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn
