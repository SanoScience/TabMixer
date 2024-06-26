from src.models.modules import DAFT, FiLM, TabAttention, TabMixer


def create_tabular_config_dict(module, tab_dim, channel_dim, frame_dim, h_dim, w_dim, additional_args=None):
    if additional_args is None:
        return {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
    else:
        conf = {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
        return {**conf, **additional_args}


def get_tabular_config(input_shape, model_name, module, tab_dim, additional_args=None):
    b, c, f, h, w = input_shape
    if model_name == "ResNet18":
        return [
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=64, frame_dim=f, h_dim=h // 2,
                                       w_dim=w // 2, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=64, frame_dim=f, h_dim=h // 2,
                                       w_dim=w // 2, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=128, frame_dim=f // 2, h_dim=h // 4,
                                       w_dim=w // 4, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=256, frame_dim=f // 4, h_dim=h // 8,
                                       w_dim=w // 8, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=512, frame_dim=f // 8, h_dim=h // 16,
                                       w_dim=w // 16, additional_args=additional_args),
        ]
    elif model_name == "Inception":
        return [
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=64, frame_dim=f // 2, h_dim=h // 2,
                                       w_dim=w // 2, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=192, frame_dim=f // 2, h_dim=h // 4,
                                       w_dim=w // 4, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=480, frame_dim=f // 2, h_dim=h // 8,
                                       w_dim=w // 8, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=832, frame_dim=f // 4, h_dim=h // 16,
                                       w_dim=w // 16, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=1024, frame_dim=f // 8,
                                       h_dim=h // 32, w_dim=w // 32, additional_args=additional_args),
        ]
    elif model_name == "SwinTransformer":
        return [
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=96, frame_dim=f // 2, h_dim=h // 4,
                                       w_dim=w // 4, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=192, frame_dim=f // 2, h_dim=h // 8,
                                       w_dim=w // 8, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=384, frame_dim=f // 2, h_dim=h // 16,
                                       w_dim=w // 16, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=768, frame_dim=f // 2, h_dim=h // 32,
                                       w_dim=w // 32, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=768, frame_dim=f // 2, h_dim=h // 32,
                                       w_dim=w // 32, additional_args=additional_args),
        ]
    elif model_name == "MLP3D":
        return [
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=64, frame_dim=f // 4, h_dim=h // 4,
                                       w_dim=w // 4, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=64, frame_dim=f // 4, h_dim=h // 4,
                                       w_dim=w // 4, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=128, frame_dim=f // 4, h_dim=h // 8,
                                       w_dim=w // 8, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=320, frame_dim=f // 4, h_dim=h // 16,
                                       w_dim=w // 16, additional_args=additional_args),
            create_tabular_config_dict(module=module, tab_dim=tab_dim, channel_dim=512, frame_dim=f // 4, h_dim=h // 32,
                                       w_dim=w // 32, additional_args=additional_args),
        ]
    else:
        raise NotImplementedError(f"No Implementation for model {model_name}")


def get_module_from_config(conf):
    if conf["module"] == "DAFT":
        return DAFT(**conf)
    elif conf["module"] == "FiLM":
        return FiLM(**conf)
    elif conf["module"] == "TabAttention":
        return TabAttention(**conf)
    elif conf["module"] == "TabMixer":
        return TabMixer(**conf)
    else:
        raise NotImplementedError(
            f"There is no implementation of: {conf.module}")
