# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import mlflow
from composer.loggers import Logger
from composer.utils import dist, parse_uri
from mlflow.data import (
    delta_dataset_source,
    http_dataset_source,
    huggingface_dataset_source,
    uc_volume_dataset_source,
)
from omegaconf import MISSING, DictConfig, ListConfig, MissingMandatoryValue
from omegaconf import OmegaConf as om
from transformers import PretrainedConfig

from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.registry import config_transforms

log = logging.getLogger(__name__)

__all__ = [
    'pop_config',
    'calculate_batch_size_info',
    'update_batch_size_info',
    'process_init_device',
    'log_config',
    'log_dataset_uri',
]


@dataclass
class EvalConfig:
    # Eval Config required parameters:
    models: list[dict[str, Any]] = MISSING
    max_seq_len: int = MISSING
    device_eval_batch_size: Union[int, float] = MISSING

    # Eval Config optional parameters:
    code_paths: Optional[list[str]] = None

    # Eval hyperparameters
    eval_gauntlet: Optional[dict[str, Any]] = None
    eval_gauntlet_str: Optional[str] = None
    eval_loader: Optional[dict[str, Any]] = None
    eval_loaders: Optional[list[dict[str, Any]]] = None
    eval_subset_num_batches: int = -1
    icl_subset_num_batches: Optional[int] = None
    # One of icl_tasks or icl_tasks_str must be specified
    icl_tasks: Optional[list[dict[str, Any]]] = None
    icl_tasks_str: Optional[str] = None

    # Logging parameters
    python_log_level: Optional[str] = 'debug'
    loggers: Optional[dict[str, Any]] = None
    console_log_interval: Union[int, str] = '1ba'
    log_config: bool = True

    # Model/run parameters
    seed: int = 17
    precision: str = 'amp_bf16'
    run_name: Optional[str] = None
    metadata: Optional[dict[str, str]] = None

    # Distributed parameters
    dist_timeout: Union[float, int] = 600.0
    fsdp_config: Optional[dict[str, Any]] = None

    # Callback parameters
    callbacks: Optional[dict[str, Any]] = None

    # Variables to ignore
    variables: Optional[dict[str, Any]] = None


EVAL_CONFIG_KEYS = {field.name for field in fields(EvalConfig)}


@dataclass
class TrainConfig:
    """Dataclass for training configuration."""

    # Mandatory model training parameters
    model: dict[str, Any] = MISSING
    tokenizer: dict[str, Any] = MISSING
    optimizer: dict[str, Any] = MISSING
    scheduler: dict[str, Any] = MISSING
    train_loader: dict[str, Any] = MISSING
    device_train_batch_size: Union[int, float] = MISSING
    device_eval_batch_size: Union[int, float] = MISSING
    max_duration: Union[int, str] = MISSING
    max_seq_len: int = MISSING

    # Seed
    seed: int = 17

    # Precision
    precision: str = 'amp_bf16'

    # Code paths to import
    code_paths: Optional[list[str]] = None

    # Cuda allocation configuration
    max_split_size_mb: Optional[int] = None
    expandable_segments: bool = True
    cuda_load_lazy: bool = False

    # Distributed training parameters
    dist_timeout: Union[int, float] = 600.0
    fsdp_config: Optional[dict[str, Any]] = None
    tp_config: Optional[dict[str, Any]] = None
    accumulate_train_batch_on_tokens: bool = True

    # Evaluation parameters
    eval_interval: Union[int, str] = 1
    eval_loader: Optional[dict[str, Any]] = None
    eval_loaders: Optional[list[dict[str, Any]]
                          ] = None  # should not be set by the user
    icl_tasks: Optional[list[dict[str, Any]]] = None
    icl_tasks_str: Optional[str] = None  # should not be set by the user
    eval_gauntlet: Optional[dict[str, Any]] = None
    eval_gauntlet_str: Optional[str] = None  # should not be set by the user
    icl_subset_num_batches: Optional[int] = None
    icl_seq_len: Optional[int] = None

    # Logging
    loggers: Optional[dict[str, Any]] = None
    progress_bar: bool = False
    log_to_console: bool = True
    python_log_level: Optional[str] = 'debug'
    console_log_interval: Union[int, str] = '1ba'
    log_config: bool = True

    # Callbacks
    callbacks: Optional[dict[str, Any]] = None
    algorithms: Optional[dict[str, Any]] = None

    # Checkpoints
    save_folder: Optional[str] = None
    save_latest_filename: Optional[str] = None
    save_overwrite: bool = False
    save_weights_only: bool = False
    save_filename: Optional[str] = None
    save_interval: Union[str, int] = '1000ba'
    save_num_checkpoints_to_keep: int = -1
    load_path: Optional[str] = None
    load_weights_only: bool = False
    load_strict_model_weights: bool = True
    load_ignore_keys: Optional[list[str]] = None
    save_ignore_keys: Optional[list[str]] = None
    only_hf_checkpoint: bool = False
    only_composer_checkpoint: bool = False

    # Dataloader
    train_subset_num_batches: int = -1
    device_train_microbatch_size: Union[str, int, float] = 'auto'
    global_train_batch_size: Optional[int] = None
    spin_dataloaders: bool = True

    # Eval dataloader
    eval_subset_num_batches: int = -1
    eval_first: bool = False
    compile_config: Optional[dict[str, Any]] = None

    # Metadata
    metadata: Optional[dict[str, Any]] = None
    flatten_metadata: bool = True
    run_name: Optional[str] = None

    # Resumption
    autoresume: bool = False

    # Profiling
    profiler: Optional[dict[str, Any]] = None

    # Variables to ignore
    variables: Optional[dict[str, Any]] = None

    # Fields created by `update_batch_size_info`
    n_gpus: int = MISSING
    device_train_grad_accum: Union[str, int] = MISSING


TRAIN_CONFIG_KEYS = {field.name for field in fields(TrainConfig)}


def forbid_config_key(cfg_dict: dict[str, Any], key: str):
    if key in cfg_dict:
        raise ValueError(
            f'Config key `{key}` should not be set. Please remove it from the config.',
        )


def to_dict_container(cfg: Union[DictConfig, dict[str, Any]]) -> dict[str, Any]:
    maybe_dict = to_container(cfg)
    if isinstance(maybe_dict, dict):
        return maybe_dict
    else:
        raise ValueError(f'Expected a dict-like type, got {type(maybe_dict)}')


def to_list_container(
    cfg: Union[ListConfig, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    maybe_list = to_container(cfg)
    if isinstance(maybe_list, list):
        return maybe_list
    else:
        raise ValueError(f'Expected a list-like type, got {type(maybe_list)}')


def to_container(
    cfg: Optional[Union[DictConfig, ListConfig, dict[str, Any],
                        list[dict[str, Any]]]],
) -> Union[dict[str, Any], list[dict[str, Any]]]:
    """Converts a DictConfig or ListConfig to a dict or list.

    `omegaconf.to_container` does not handle nested DictConfig or ListConfig
    objects, so this function is used to convert them to dicts or lists.
    """
    if isinstance(cfg, DictConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, dict)
        return ret  # type: ignore (return type is correct and converting all keys to str would be unnecessarily costly)
    elif isinstance(cfg, ListConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, list)
        return ret  # type: ignore (see above)
    else:
        return cfg  # type: ignore (dicts and lists are already in the correct format)


T = TypeVar('T')


def apply_transforms_to_config(
    cfg: dict[str, Any],
    transforms: Optional[Union[list[Callable[[dict[str, Any]], dict[str, Any]]],
                               list[str], str]],
) -> dict[str, Any]:
    """Applies a list of transforms to a config.

    Args:
        cfg (Dict[str, Any]): The config to transform.
        transforms (Optional[Union[List[Callable[[Dict[str, Any]], Dict[str, Any]]], List[str], str]]): A list of
            transform functions or strings representing transform functions to apply to the config. If a single string
            with the value ``all`` is provided, all registered transforms will be applied.

    Returns:
        Dict[str, Any]: The transformed config.
    """
    if transforms is None or (
        isinstance(transforms, list) and len(transforms) == 0
    ):
        return cfg

    transform_functions = []
    if isinstance(transforms, list):
        for transform in transforms:
            if isinstance(transform, str):
                transform_functions.append(config_transforms.get(transform))
            elif callable(transform):
                transform_functions.append(transform)
            else:
                raise ValueError(
                    f'Invalid transform: {transform}. Must be a string or callable.',
                )
    elif isinstance(transforms, str) and transforms == 'all':
        transform_functions = [
            config_transforms.get(transform)
            for transform in config_transforms.get_all()
        ]
    else:
        raise ValueError(
            f'Invalid transforms: {transforms}. Must be a list of strings or callables, or ``all``.',
        )

    for transform in transform_functions:
        cfg = transform(cfg)
    return cfg


def make_dataclass_and_log_config(
    cfg: DictConfig,
    dataclass_constructor: Callable[..., T],
    dataclass_fields: set[str],
    transforms: Optional[Union[list[Callable[[dict[str, Any]], dict[str, Any]]],
                               list[str], str]] = None,
    icl_tasks_required: bool = False,
) -> tuple[dict[str, Any], T]:
    """Converts a DictConfig to a dataclass and creates a logged config."""
    unstructured_config = om.to_container(cfg, resolve=True)
    assert isinstance(unstructured_config, dict)
    assert all(isinstance(k, str) for k in unstructured_config.keys())
    unstructured_config = {str(k): v for k, v in unstructured_config.items()}

    # Flatten union types before creating structured config:
    if 'eval_gauntlet' in unstructured_config:
        forbid_config_key(unstructured_config, 'eval_gauntlet_str')
        if isinstance(unstructured_config['eval_gauntlet'], str):
            unstructured_config['eval_gauntlet_str'] = unstructured_config.pop(
                'eval_gauntlet',
            )
    if (loader := unstructured_config.get('eval_loader', None)) is not None:
        forbid_config_key(unstructured_config, 'eval_loaders')
        if isinstance(loader, list):
            unstructured_config['eval_loaders'] = unstructured_config.pop(
                'eval_loader',
            )
    if 'icl_tasks' in unstructured_config:
        forbid_config_key(unstructured_config, 'icl_tasks_str')
        if isinstance(unstructured_config['icl_tasks'], str):
            unstructured_config['icl_tasks_str'] = unstructured_config.pop(
                'icl_tasks',
            )
    else:
        if icl_tasks_required:
            raise MissingMandatoryValue(
                'icl_tasks must be specified in the config',
            )

    # Apply transforms to the unstructured config before constructing dataclass
    unstructured_config = apply_transforms_to_config(
        unstructured_config,
        transforms,
    )

    # Create copy of config for logging
    logged_cfg: dict[str, Any] = copy.deepcopy(unstructured_config)

    arg_config_keys = set(unstructured_config.keys())
    extraneous_keys = set.difference(arg_config_keys, dataclass_fields)

    if 'variables' not in unstructured_config:
        unstructured_config['variables'] = {}

    if len(extraneous_keys) > 0:
        print('Final config keys:', list(cfg.keys()))
        raise ValueError(
            f'Unused parameters {sorted(extraneous_keys)} found in cfg. Please check your yaml to ensure these parameters are necessary. Please place any variables under the `variables` key.',
        )

    dataclass_dict_config: DictConfig = om.structured(
        dataclass_constructor(**unstructured_config),
    )

    # Error on missing mandatory values:
    for key in dataclass_fields:
        _ = dataclass_dict_config[key]

    # Convert DictConfig to dict for dataclass constructor so that child
    # configs are not DictConfigs
    dataclass_config: T = dataclass_constructor(
        **to_dict_container(dataclass_dict_config),
    )

    return logged_cfg, dataclass_config


def pop_config(
    cfg: Union[dict[str, Any], DictConfig],
    key: str,
    must_exist: bool = True,
    default_value: Any = None,
    convert: bool = False,
) -> Any:
    """Pop a value from the main config file and return it.

    If the key does not exist, return the default_value or raise a RuntimeError
    depending on the must_exist flag. If the convert flag is set to True, then
    we will convert the value to a python object using OmegaConf.to_container.
    """
    value = cfg.pop(key, None)
    if value is not None and convert:
        if not isinstance(value,
                          DictConfig) and not isinstance(value, ListConfig):
            raise ValueError(
                f'The key {key} has a value of type {type(value)} that cannot be \
                            converted to a dict or list. Please check your yaml.',
            )
        return om.to_container(value)
    elif value is not None:
        return value
    elif must_exist:
        raise NameError(
            f'The {key} parameter is missing and must exist for execution. Please check your yaml.',
        )
    else:
        return default_value


def get_hf_config_value(config: Union[dict, PretrainedConfig], key: str) -> Any:
    """Get a value from a Hugging Face config.

    Args:
        config (Union[dict, PretrainedConfig]): The Hugging Face config object.
        key (str): The key to get from the config.

    Returns:
        Any: The value from the config. None if the key does not exist.
    """
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def calculate_batch_size_info(
    global_batch_size: int,
    device_microbatch_size: Union[int, float, Literal['auto']],
    data_replication_degree: int = 1,
) -> tuple[Union[int, float], Union[int, float, Literal['auto']], Union[
    int, Literal['auto']]]:

    world_size = dist.get_world_size()
    if world_size % data_replication_degree != 0:
        raise ValueError(
            f'World size {world_size} is not divisible by data replication degree {data_replication_degree}.',
        )
    if global_batch_size % (world_size // data_replication_degree) != 0:
        raise ValueError(
            f'Global batchsize {global_batch_size} is not divisible by {(world_size // data_replication_degree)=} '
            +
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            + f'to be divisible by world size, {world_size}.',
        )
    device_batch_size = global_batch_size / world_size
    if device_batch_size == round(device_batch_size):
        device_batch_size = round(device_batch_size)
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, (int, float)):
        if device_microbatch_size > device_batch_size:
            log.warning(
                f'device_microbatch_size > device_batch_size, ' +
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.',
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(
            device_batch_size / device_microbatch_size,  # type: ignore
        )
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


def update_config_with_batch_size_info(
    cfg: dict[str, Any],
    device_train_batch_size: Union[int, float],
    device_train_microbatch_size: Union[int, float, Literal['auto']],
    device_train_grad_accum: Union[int, Literal['auto']],
) -> dict[str, Any]:
    """Update the config with batch size information.

    Args:
        cfg (Dict[str, Any]): The config to update.
        device_train_batch_size (Union[int, float]): The batch size of the training dataset for each device.
        device_train_microbatch_size (Union[int, float, Literal['auto']]): The microbatch size of the training dataset for each device.
        device_train_grad_accum (Union[int, Literal['auto']]): The gradient accumulation settings for each device.

    Returns:
        Dict[str, Any]: The updated config.
    """
    cfg['n_gpus'] = dist.get_world_size()
    cfg['device_train_batch_size'] = device_train_batch_size
    cfg['device_train_microbatch_size'] = device_train_microbatch_size
    cfg['device_train_grad_accum'] = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg['device_train_microbatch_size'] == 'auto':
            cfg['device_eval_batch_size'
               ] = 1  # TODO debug auto eval microbatching
        else:
            cfg['device_eval_batch_size'] = cfg['device_train_microbatch_size']
    return cfg


def update_batch_size_info(cfg: dict[str, Any]) -> dict[str, Any]:
    data_replication_degree = 1
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg['global_train_batch_size'],
        cfg['device_train_microbatch_size'],
        data_replication_degree=data_replication_degree,
    )
    cfg = update_config_with_batch_size_info(
        cfg,
        device_train_batch_size,
        device_train_microbatch_size,
        device_train_grad_accum,
    )
    return cfg


def process_init_device(
    model_cfg: dict[str, Any],
    fsdp_config: Optional[dict] = None,
    tp_config: Optional[dict] = None,
):
    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in model_cfg:
        assert model_cfg['init_device'] in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and model_cfg['init_device'] == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            model_cfg['init_device'] = 'cpu'
        if model_cfg['init_device'] == 'meta':
            init_context = init_empty_weights()
        if model_cfg['init_device'] == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. ' +
                    'Please add a FSDP config.',
                )
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'
                ))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('load_monolith_rank0_only', True)

    if tp_config is not None:
        # Check tp_config has required fields
        if 'strategy' not in tp_config or 'tensor_parallel_degree' not in tp_config:
            raise ValueError(
                "`tp_config` requires 'strategy' and 'tensor_parallel_degree' values. ",
            )

        # Check we are not using tensor parallelism with MoEs
        if 'ffn_config' in model_cfg and model_cfg['ffn_config'].get(
            'ffn_type',
            None,
        ) in ffns_with_megablocks:
            raise ValueError(
                'Tensor Parallelism is not currently supported for MoE models.',
            )

    # Check we are not using tensor parallelism with MoEs
    if tp_config is not None and 'ffn_config' in model_cfg and model_cfg[
        'ffn_config'].get('ffn_type', None) in ffns_with_megablocks:
        raise ValueError(
            'Tensor Parallelism is not currently supported for MoE models.',
        )

    # Set ffn_config.device_mesh using fsdp_config
    if fsdp_config is not None and 'ffn_config' in model_cfg and model_cfg[
        'ffn_config'].get('ffn_type', None) in ffns_with_megablocks:
        shard_degree = fsdp_config.get('data_parallel_shard_degree', None)
        replicate_degree = fsdp_config.get(
            'data_parallel_replicate_degree',
            None,
        )

        if shard_degree is None and replicate_degree is None:
            # Default to sharding over all gpus.
            shard_degree = dist.get_world_size()
            device_mesh_cfg = [shard_degree]
        else:
            if shard_degree is None:
                # Shard degree is not specified, so calculate it from replicate degree
                assert isinstance(replicate_degree, int)
                shard_degree = dist.get_world_size() // replicate_degree
            elif replicate_degree is None:
                # Replicate degree is not specified, so calculate it from shard degree
                assert isinstance(shard_degree, int)
                replicate_degree = dist.get_world_size() // shard_degree

            if replicate_degree == 1:
                device_mesh_cfg = [shard_degree]
            else:
                device_mesh_cfg = [replicate_degree, shard_degree]

        model_cfg['ffn_config']['device_mesh'] = device_mesh_cfg

    # No mixed precision needed for weights when they're already 16 bits
    master_dtype = model_cfg.get('master_weights_dtype')
    small_dtypes = (
        'bf16',
        'fp16',
        'float16',
        'bfloat16',
        'amp_fp16',
        'amp_bf16',
    )
    if fsdp_config and master_dtype in small_dtypes:
        reduce_dtype = None
        buffer_dtype = None
        mixed_precision = fsdp_config.get('mixed_precision')
        if isinstance(mixed_precision, Mapping):
            reduce_dtype = mixed_precision.get('reduce_dtype')
            buffer_dtype = mixed_precision.get('buffer_dtype')
        fsdp_config['mixed_precision'] = {
            'param_dtype': None,
            'reduce_dtype': reduce_dtype,
            'buffer_dtype': buffer_dtype,
            'keep_low_precision_grads': True,
        }

    return init_context


def log_config(logger: Logger, cfg: dict[str, Any]) -> None:
    """Logs the current config and updates the wandb and mlflow configs.

    This function can be called multiple times to update the wandb and MLflow
    config with different variables.
    """
    print(om.to_yaml(cfg))
    logger.log_hyperparameters(cfg)


def _parse_source_dataset(cfg: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Parse a run config for dataset information.

    Given a config dictionary, parse through it to determine what the datasource
    should be categorized as. Possible data sources are Delta Tables, UC Volumes,
    HuggingFace paths, remote storage, or local storage.

    Args:
        cfg (DictConfig): A config dictionary of a run

    Returns:
        List[Tuple[str, str, str]]: A list of tuples formatted as (data type, path, split)
    """
    data_paths = []

    # Handle train loader if it exists
    train_dataset: dict = cfg.get('train_loader', {}).get('dataset', {})
    train_split = train_dataset.get('split', None)
    train_source_path = cfg.get('source_dataset_train', None)
    _process_data_source(
        train_source_path,
        train_dataset,
        train_split,
        'train',
        data_paths,
    )

    # Handle eval_loader which might be a list or a single dictionary
    eval_data_loaders = cfg.get('eval_loader', {})
    if not isinstance(eval_data_loaders, list):
        eval_data_loaders = [
            eval_data_loaders,
        ]  # Normalize to list if it's a single dictionary

    for eval_data_loader in eval_data_loaders:
        assert isinstance(eval_data_loader, dict)  # pyright type check
        eval_dataset: dict = eval_data_loader.get('dataset', {})
        eval_split = eval_dataset.get('split', None)
        eval_source_path = cfg.get('source_dataset_eval', None)
        _process_data_source(
            eval_source_path,
            eval_dataset,
            eval_split,
            'eval',
            data_paths,
        )

    return data_paths


def _process_data_source(
    source_dataset_path: Optional[str],
    dataset: dict[str, str],
    cfg_split: Optional[str],
    true_split: str,
    data_paths: list[tuple[str, str, str]],
):
    """Add a data source by mutating data_paths.

    Given various dataset attributes, attempt to determine what type of dataset is being added, and parse
    the dataset accordingly.

    Args:
        source_dataset_path (Optional[str]): The source dataset in cfg metadata
        dataset (Dict[str, str]): The dataset from cfg
        cfg_split (str): The split listed for the dataset in cfg
        true_split (str): The split of the dataset to be added (i.e. train or eval)
        data_paths (List[Tuple[str, str, str]]): A list of tuples formatted as (data type, path, split)
    """
    if source_dataset_path:
        source_dataset_path = str(Path(source_dataset_path))
    # Check for Delta table
    if source_dataset_path and len(source_dataset_path.split('.')) == 3:
        data_paths.append(('delta_table', source_dataset_path, true_split))
    # Check for UC volume
    elif source_dataset_path and source_dataset_path.startswith('dbfs:'):
        data_paths.append(
            ('uc_volume', source_dataset_path[len('dbfs:'):], true_split),
        )
    # Check for HF path
    elif 'hf_name' in dataset and dataset['hf_name']:
        hf_path = dataset['hf_name']
        backend, _, uc_path = parse_uri(hf_path)
        unsupported_file = True
        if backend == 'dbfs':
            assert cfg_split
            from llmfoundry.data.finetuning.tasks import SUPPORTED_EXTENSIONS
            possible_files = [
                f'{cfg_split}{ext}' for ext in SUPPORTED_EXTENSIONS
            ]
            for file in possible_files:
                path = os.path.join(uc_path, file)
                # Ensure path starts with '/'
                if not path.startswith('/'):
                    path = '/' + path
                if _verify_uc_path(path):
                    data_paths.append(('uc_volume', path, true_split))
                    unsupported_file = False
                    break
            if unsupported_file:
                log.warning(
                    f'{hf_path} does not contain a supported file extension.',
                )
        elif backend:
            hf_path = os.path.join(hf_path, cfg_split) if cfg_split else hf_path
            data_paths.append((backend, hf_path, true_split))
        elif os.path.exists(hf_path):
            data_paths.append(('local', hf_path, true_split))
        else:
            data_paths.append(('hf', hf_path, true_split))
    # Check for remote path
    elif 'remote' in dataset and dataset['remote']:
        remote_path = dataset['remote']
        backend, _, _ = parse_uri(remote_path)
        if backend:
            remote_path = os.path.join(
                remote_path,
                f'{cfg_split}/',
            ) if cfg_split else remote_path
            data_paths.append((backend, remote_path, true_split))
        else:
            # No backend detected so assume local path
            data_paths.append(('local', remote_path, true_split))
    # Check for local path
    elif 'local' in dataset and dataset['local']:
        data_paths.append(('local', dataset['local'], true_split))
    else:
        log.warning('DataSource Not Found.')


def log_dataset_uri(cfg: dict[str, Any]) -> None:
    """Logs dataset tracking information to MLflow.

    Args:
        cfg (DictConfig): A config dictionary of a run
    """
    loggers = cfg.get('loggers', None) or {}
    if 'mlflow' not in loggers or not mlflow.active_run():
        return
    # Figure out which data source to use
    data_paths = _parse_source_dataset(cfg)

    dataset_source_mapping = {
        's3': http_dataset_source.HTTPDatasetSource,
        'oci': http_dataset_source.HTTPDatasetSource,
        'azure': http_dataset_source.HTTPDatasetSource,
        'gs': http_dataset_source.HTTPDatasetSource,
        'https': http_dataset_source.HTTPDatasetSource,
        'hf': huggingface_dataset_source.HuggingFaceDatasetSource,
        'delta_table': delta_dataset_source.DeltaDatasetSource,
        'uc_volume': uc_volume_dataset_source.UCVolumeDatasetSource,
        'local': http_dataset_source.HTTPDatasetSource,
    }

    # Map data source types to their respective MLFlow DataSource.
    for dataset_type, path, split in data_paths:
        if dataset_type in dataset_source_mapping:
            source_class = dataset_source_mapping[dataset_type]
            if dataset_type == 'delta_table':
                source = source_class(delta_table_name=path)
            elif dataset_type == 'hf' or dataset_type == 'uc_volume':
                source = source_class(path=path)
            else:
                source = source_class(url=path)
        else:
            log.info(
                f'{dataset_type} unknown, defaulting to http dataset source',
            )
            source = http_dataset_source.HTTPDatasetSource(url=path)

        mlflow.log_input(
            mlflow.data.meta_dataset.MetaDataset(source, name=split),
        )


def _verify_uc_path(path: str) -> bool:
    """Verify a UC path exists.

    Args:
        path (str): UnityCatalog path
    Returns:
        (bool): If path exists or not
    """
    from databricks.sdk.errors.platform import NotFound, PermissionDenied
    w = None
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
    except ImportError:
        log.warning(
            'Cannot verify the path of `UCVolumeDatasetSource` because of missing' + \
            '`databricks-sdk`. Please install `databricks-sdk` via ' + \
            '`pip install -U databricks-sdk`. This does not block creating ' + \
            '`UCVolumeDatasetSource`, but your `UCVolumeDatasetSource` might be invalid.',
        )
        return False
    except Exception as e:
        log.warning(
            f'Error occured when attempting to connect with Databricks WorkspaceClient. ' + \
            f'Error details: {str(e)}. This does not block creating `UCVolumeDatasetSource`, ' + \
            f'but your `UCVolumeDatasetSource` might be invalid.',
        )

    if w:
        try:
            w.files.get_metadata(path)
        except (NotFound, PermissionDenied):
            try:
                # Check if `self.path` points to a valid UC directory.
                w.files.get_directory_metadata(path)
                return True
            except (NotFound, PermissionDenied):
                # Neither file nor directory exists, we throw an exception.
                return False
        except Exception as e:
            log.warning(
                f'Error occured when verifying path of `UCVolumeDatasetSource`. ' + \
                f'Error details: {str(e)}. This does not block creating `UCVolumeDatasetSource`, ' + \
                f'but your `UCVolumeDatasetSource` might be invalid.',
            )
    return False


def set_config_overrides(
    config: PretrainedConfig,
    config_overrides: dict[str, Any],
):
    # set config overrides
    for k, v in config_overrides.items():
        if not hasattr(config, k):
            raise ValueError(
                f'config does not have attribute "{k}" to override ({k}: {v}).',
            )

        attr = getattr(config, k)
        # attempt to disallow typos in nested configs
        if isinstance(attr, Mapping):
            extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
            if extra_keys:
                raise ValueError(
                    f'Config dict override got unknown keys. ' +
                    f'Extra keys: {extra_keys}. ' +
                    f'Expected (a subset of) keys: {list(attr.keys())}.',
                )
            getattr(config, k).update(v)
        # necessary case to allow for rope_scaling to be overriden in llama config
        elif attr is None and isinstance(v, Mapping):
            setattr(config, k, {})
            getattr(config, k).update(v)
        elif isinstance(attr, PretrainedConfig):
            if not isinstance(v, Mapping):
                raise ValueError(
                    f'Expected a dictionary for config override {k}, but got {v}.',
                )

            for _k, _v in v.items():
                if not hasattr(attr, _k):
                    raise ValueError(
                        f'config does not have attribute "{_k}" to override ({k}: {_k}: {_v}).',
                    )
                setattr(attr, _k, _v)
        else:
            setattr(config, k, v)
