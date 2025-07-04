# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Generator, Optional, Union

import torch
from torch import nn
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Placement,
    distribute_tensor,
)

from llmfoundry.layers_registry import (
    fcs,
    module_init_fns,
    norms,
    param_init_fns,
)
from llmfoundry.models.layers.dmoe import GLU, MLP

try:
    import transformer_engine.pytorch as te
except:
    te = None

try:
    import megablocks
except:
    megablocks = None

__all__ = [
    'generic_param_init_fn_',
]


@contextmanager
def materialize_tensor(
    tensor: torch.Tensor,
) -> Generator[torch.Tensor, None, None]:
    """Context manager for initializing DTensor parameters.

    This context manager temporarily materializes a DTensor as a full tensor for initialization,
    then redistributes the initialized values back to the original DTensor.

    NOTE: This approach is not memory-efficient as it materializes a full tensor on every rank
    and then redistributes it back to a DTensor. However, since model initialization happens
    only once, this overhead is generally acceptable.

    Args:
        tensor: The tensor to materialize. Can be either a regular Tensor or a DTensor.

    Yields:
        torch.Tensor: The materialized full tensor that can be initialized.
    """
    is_dtensor = isinstance(tensor, DTensor)

    with torch.no_grad():
        if is_dtensor:
            full_tensor = tensor.full_tensor()
        else:
            full_tensor = tensor

        yield full_tensor
        if is_dtensor:
            # Redistribute the updated full tensor back to the original DTensor
            with torch.no_grad():
                temp_tensor = distribute_tensor(
                    full_tensor,
                    device_mesh=tensor.device_mesh,
                    placements=tensor.placements,
                )
                tensor.to_local().copy_(temp_tensor.to_local())


@contextmanager
def materialize_module(
    module: nn.Module,
) -> Generator[nn.Module, None, None]:
    """Context manager for initializing modules containing DTensor parameters.

    This context manager temporarily materializes all DTensor parameters in a module
    as full tensors for initialization, then redistributes the initialized values back
    to the original DTensor parameters.

    Args:
        module: The module whose parameters should be materialized.

    Yields:
        nn.Module: The module with materialized parameters that can be initialized.
    """
    param_placement_map: dict[tuple[nn.Module, str],
                              tuple[DeviceMesh, tuple[Placement, ...]]] = {}
    with torch.no_grad():
        for submodule in module.modules():
            for name, param in submodule.named_parameters(recurse=False):
                if isinstance(param, DTensor):
                    param_placement_map[
                        (submodule,
                         name)] = (param.device_mesh, param.placements)
                    setattr(submodule, name, nn.Parameter(param.full_tensor()))

        yield module

        for submodule in module.modules():
            for name, param in submodule.named_parameters(recurse=False):
                if (submodule, name) in param_placement_map:
                    device_mesh, placements = param_placement_map[
                        (submodule, name)]
                    setattr(
                        submodule,
                        name,
                        nn.Parameter(
                            distribute_tensor(
                                param,
                                device_mesh=device_mesh,
                                placements=placements,
                            ),
                        ),
                    )


def summon_dtensor(init_fn: Callable) -> Callable:
    """Unshard and Reshard DTensor parameters for initialization.

    This decorator wraps an initialization function to handle both regular tensors/modules
    and those containing DTensor parameters by temporarily materializing DTensors as full
    tensors during initialization.

    Args:
        init_fn: The initialization function to wrap.

    Returns:
        A wrapped initialization function that can handle DTensor parameters.
    """

    def init_fn_wrapper(
        obj: nn.Module | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if isinstance(obj, nn.Module):
            with materialize_module(obj) as obj:
                return init_fn(obj, *args, **kwargs)
        elif isinstance(obj, torch.Tensor):
            with materialize_tensor(obj) as obj:
                return init_fn(obj, *args, **kwargs)
        else:
            raise TypeError(f'Invalid object type: {type(obj)}')

    return init_fn_wrapper


def torch_default_param_init_fn_(
    module: nn.Module,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

    if hasattr(
        module,
        'reset_parameters',
    ) and isinstance(module.reset_parameters, Callable):
        module.reset_parameters()


def fused_init_helper_(
    module: nn.Module,
    init_fn_: Callable,
    name_param: str = 'weight',
):
    """Initializes parameters which have been fused for efficiency purposes.

    Parameter initialization is often based on the parameters shape. If a layer is fused,
    initialization should be based on the shapes of the original tensor instead of the
    shape of the fused tensor. Layers which are fused should have the _fused
    attribute. First element of _fused is the dimension along which the tensor is fused.
    Second element is a an iterable of split indices.

    Args:
        module (nn.Module): The module to initialize.
        init_fn_ (Callable): Initialization method.
        name_param (str): Name of parameter to initialize within the module.
    """
    _fused = getattr(module, '_fused', None)
    if _fused is None:
        raise RuntimeError(f'Internal logic error')

    fused_param_init_helper(getattr(module, name_param), init_fn_, _fused)


@summon_dtensor
def fused_param_init_helper(
    param: torch.Tensor,
    init_fn_: Callable,
    fused_parameters: tuple[int, list[int]],
):
    """Initializes parameters that are fused together.

    Args:
        param (torch.Tensor): Tensor to initialize.
        init_fn_ (Callable): Initialization method.
        fused_parameters (tuple[int, list[int]]): First element of _fused is the dimension
            along which the tensor is fused. Second element is a an iterable of split indices.
    """
    p_ndims = param.ndim
    dim, splits = fused_parameters
    splits = (0, *splits, param.size(dim))  # type: ignore
    for s, e in zip(splits[:-1], splits[1:]):
        # DTensor slicing results in CC and thus produces new Tensor
        # so the update is not inplace, additionally, the init_fn is
        # designed for full tensors, not for a (sharded) DTensor, so
        # we need this context manager to handle the DTensor case
        slice_indices = [slice(None)] * p_ndims  # type: ignore
        slice_indices[dim] = slice(s, e)
        init_fn_(param[slice_indices])  # type: ignore


def stacked_init_helper_(
    module: nn.Module,
    init_fn_: Callable,
    name_param: str = 'weight',
):
    """Initializes parameters stacked along a new dimension.

    Parameter initialization is often based on the parameters shape. If a layer is stacked,
    initialization should be based on the shapes of the original tensor instead of the
    shape of the stacked tensor. Layers which are fused should have the _stacked_dim
    attribute defining the new dimension along which they are stacked.

    Args:
        module (nn.Module): The module to initialize.
        init_fn_ (Callable): Initialization method.
        name_param (str): Name of parameter to initialize within the module.
    """
    stack_dim = getattr(module, '_stack_dim', None)
    if stack_dim is None:
        raise RuntimeError(f'Internal logic error')

    stacked_param_init_helper(getattr(module, name_param), init_fn_, stack_dim)


@summon_dtensor
def stacked_param_init_helper(
    param: torch.Tensor,
    init_fn_: Callable,
    stack_dim: int,
):
    """Initialize parameters stacked along a new dimension.

    Args:
        param (torch.Tensor): Tensor to initialize.
        init_fn_ (Callable): Initialization method.
        stack_dim (int): Dimension along with parameters are stacked
    """
    p_ndims = param.ndim

    for idx in range(param.size(stack_dim)):
        slice_indices = [slice(None)] * p_ndims  # type: ignore
        slice_indices[stack_dim] = idx  # type: ignore
        init_fn_(param[slice_indices])  # type: ignore


def _flip_fan_mode(init_fn_: Callable):
    """Changes the mode of an init_fn_.

    init_fn_'s "mode" is set to operate on standard torch modules eg torch.nn.Linear.
    If a custom layer transposes its weights before they are allied such that it is
    opposite pytorch's conventions, we must flip the fan mode, from fan_in to fan_out.

    Args:
        init_fn_ (Callable): Initialization method.
    """
    _init_fn_ = deepcopy(init_fn_)
    if 'mode' in _init_fn_.keywords:
        if _init_fn_.keywords['mode'] == 'fan_in':
            _init_fn_.keywords['mode'] = 'fan_out'
        elif _init_fn_.keywords['mode'] == 'fan_out':
            _init_fn_.keywords['mode'] = 'fan_in'
    return _init_fn_


@summon_dtensor
def fc_init(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: Optional[float],
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, tuple({fcs.get(n) for n in fcs.get_all()})):
        # Linear
        if hasattr(module, '_fused'):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)
            torch.nn.init.zeros_(module.bias)

        if init_div_is_residual is not False and getattr(
            module,
            '_is_residual',
            False,
        ):
            with torch.no_grad():
                module.weight.div_(div_is_residual)  # type: ignore
        return True

    return False


@summon_dtensor
def embedding_init(
    module: nn.Module,
    init_fn_: Callable,
    emb_init_std: Optional[float],
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]],
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, nn.Embedding):
        # Embedding
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f'Embedding layer initialized to 0.')
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
        elif emb_init_uniform_lim is not None:
            lim = emb_init_uniform_lim
            if isinstance(lim, Sequence):
                if len(lim) > 2:
                    raise ValueError(
                        f'Uniform init requires a min and a max limit. User input: {lim}.',
                    )
                if lim[0] == lim[1]:
                    warnings.warn(f'Embedding layer initialized to {lim[0]}.')
            else:
                if lim == 0:
                    warnings.warn(f'Embedding layer initialized to 0.')
                lim = [-lim, lim]
            a, b = lim
            emb_init_fn_ = partial(torch.nn.init.uniform_, a=a, b=b)
        else:
            emb_init_fn_ = init_fn_

        emb_init_fn_(module.weight)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)

        return True

    return False


def norm_init(
    module: nn.Module,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(
        module,
        tuple({norms.get(name) for name in norms.get_all()}),
    ):
        # Norm
        if hasattr(module,
                   'weight') and isinstance(module.weight, torch.Tensor):
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            torch.nn.init.zeros_(module.bias)

        return True

    return False


@summon_dtensor
def multihead_attention_init(
    module: nn.Module,
    init_fn_: Callable,
    d_model: Optional[int],
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, nn.MultiheadAttention):
        # torch's MultiheadAttention
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
            assert d_model is not None
            # in_proj_weight is actually 3 layers and should be split up for width based init
            _d = d_model
            splits = (0, _d, 2 * _d, 3 * _d)
            for s, e in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
            assert module.in_proj_weight is None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)

        # bias
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)

        # out proj
        init_fn_(module.out_proj.weight)
        if init_div_is_residual is not False and getattr(
            module.out_proj,
            '_is_residual',
            False,
        ):
            with torch.no_grad():
                module.out_proj.weight.div_(div_is_residual)
        if module.out_proj.bias is not None:
            torch.nn.init.zeros_(module.out_proj.bias)

        return True

    return False

@summon_dtensor
@module_init_fns.register("sparse_attention")
def sparse_attention_init(
    module: nn.Module,
    init_fn_: Callable,
    d_model: Optional[int],
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs,
) -> bool:
    """
    Initialiser for native SparseAttention.

    We do *not* touch the module’s own parameters, because every learnable tensor
    inside SparseAttention is an nn.Linear or nn.Parameter that will already be
    handled by the 'fc' or default initialisers.  We just need to return True so
    generic_param_init_fn_ knows the module is covered.
    """
    if isinstance(module, SparseAttention):
        return True          # <- claims the module, nothing more to do
    return False


def te_layernorm_mlp_init(
    module: nn.Module,
    init_fn_: Callable,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if te is not None and isinstance(module, te.LayerNormMLP):
        if isinstance(module.layer_norm_weight, torch.Tensor):
            torch.nn.init.ones_(module.layer_norm_weight)
        if isinstance(module.layer_norm_bias, torch.Tensor):
            torch.nn.init.zeros_(module.layer_norm_bias)

        init_fn_(module.fc1_weight)
        if module.fc1_bias is not None:
            assert isinstance(module.fc1_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc1_bias)
        init_fn_(module.fc2_weight)
        if module.fc2_bias is not None:
            assert isinstance(module.fc2_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc2_bias)

        with torch.no_grad():
            module.fc2_weight.div_(div_is_residual)  # type: ignore

        return True

    return False


def moe_init(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs: Any,
) -> bool:
    if megablocks is not None and isinstance(
        module,
        (
            megablocks.layers.moe.MoE,
            megablocks.layers.dmoe.dMoE,
            megablocks.layers.moe.ParallelMLP,
            megablocks.layers.dmoe.ParallelDroplessMLP,
        ),
    ):
        if hasattr(module, 'bias') and module.bias is not None:
            # Initialize bias to 0
            torch.nn.init.zeros_(module.bias)  # type: ignore
        return True
    elif megablocks is not None and isinstance(
        module,
        megablocks.layers.glu.SparseGLU,
    ):
        _megablocks_sparse_glu_generic_param_init_fn_(
            module,
            init_fn_,
            bool(init_div_is_residual),
            div_is_residual,
        )
        return True
    elif megablocks is not None and isinstance(
        module,
        megablocks.layers.mlp.SparseMLP,
    ):
        _megablocks_sparse_mlp_generic_param_init_fn_(
            module,
            init_fn_,
            bool(init_div_is_residual),
            div_is_residual,
        )
        return True
    elif megablocks is not None and isinstance(
        module,
        megablocks.layers.mlp.MLP,
    ):
        _megablocks_mlp_generic_param_init_fn_(
            module,
            init_fn_,
            bool(init_div_is_residual),
            div_is_residual,
        )
        return True
    elif isinstance(module, GLU):
        init_fn_(module.w1)
        init_fn_(module.v1)
        init_fn_(module.w2)
        return True
    elif isinstance(module, MLP):
        init_fn_(module.w1)
        init_fn_(module.w2)
        return True

    return False


def generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    # enable user to divide _is_residual weights by

    # a value which defaults to math.sqrt(2 * cfg.n_layers)
    init_div_is_residual = init_div_is_residual

    if init_div_is_residual is False:
        # not used, for pyright
        div_is_residual = 1.0
    elif init_div_is_residual is True:
        div_is_residual = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual,
                    float) or isinstance(init_div_is_residual, int):
        div_is_residual = init_div_is_residual
    elif init_div_is_residual.isnumeric():
        # do not trust YAML parsing to always convert numbers to numbers
        div_is_residual = float(init_div_is_residual)
    else:
        # not used, for pyright
        div_is_residual = 1.0
        raise ValueError(
            f'Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}',
        )

    all_module_init_fns = [
        module_init_fns.get(name) for name in module_init_fns.get_all()
    ]
    did_init = False
    for module_init_fn in all_module_init_fns:
        did_init = module_init_fn(
            module,
            init_fn_=init_fn_,
            d_model=d_model,
            init_div_is_residual=init_div_is_residual,
            div_is_residual=div_is_residual,
            emb_init_std=emb_init_std,
            emb_init_uniform_lim=emb_init_uniform_lim,
        )

        if did_init:
            break

    if not did_init:
        for _ in module.parameters(recurse=False):
            # raise error if uninitialized module has any parameters
            raise NotImplementedError(
                f'{module.__class__.__name__} parameters are not initialized by any of the registered module_init_fns. '
                +
                'Please add an appropriate module_init_fn to the registry. Currently registered module_init_fns are: '
                + ', '.join(module_init_fns.get_all()),
            )


def _megablocks_sparse_mlp_generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: bool = False,
    div_is_residual: float = 1.0,
):
    """Initializes MegaBlocks MLP.

    To enable elastic deterministic initialization, this method creates the entire
    weight matrix then slice into the weight tensors such that the sampled weights
    should not vary between moe world size for the same random seed.

    Args:
        module (nn.Module): The module to initialize.
        init_fn_ (Callable): Initialization method.
        init_div_is_residual (bool): Flag enabling parameters tagged with _is_residual
            flag to be divided by div_is_residual.
        div_is_residual (float): The value by which parameter initialization is divided
            if init_div_is_residual flag is enabled.
    """
    expert_process_group_size, rank = 1, 0
    if module.expert_parallel_group is not None:
        expert_process_group_size = int(
            module.expert_parallel_group.size(),  # type: ignore
        )
        rank = int(module.expert_parallel_group.rank())  # type: ignore

    hidden_size = int(module.hidden_size)  # type: ignore

    # Initialize w1
    w1 = module.w1
    if isinstance(w1, DTensor):
        w1 = w1._local_tensor
    w1_size = list(w1.shape)  # type: ignore
    w1_size[0] = w1_size[0] * expert_process_group_size

    n_exp = w1_size[0] // hidden_size
    _fused = (0, [(n + 1) * hidden_size for n in range(n_exp - 1)])

    _w1 = w1.new_empty(w1_size)  # type: ignore
    fused_param_init_helper(_w1, init_fn_, _fused)
    _w1_local = _w1.chunk(expert_process_group_size, dim=0)[rank]
    with torch.no_grad():
        w1.copy_(_w1_local)  # type: ignore

    # Initialize w2
    w2 = module.w2
    if isinstance(w2, DTensor):
        w2 = w2._local_tensor
    w2_size = list(w2.shape)  # type: ignore
    w2_size[0] = w2_size[0] * expert_process_group_size
    _w2 = w2.new_empty(w2_size)  # type: ignore
    # MegaBlocks operates on w2 as x @ w2, so needs flipped fan mode
    fused_param_init_helper(_w2, _flip_fan_mode(init_fn_), _fused)
    _w2_local = _w2.chunk(expert_process_group_size, dim=0)[rank]
    with torch.no_grad():
        w2.copy_(_w2_local)  # type: ignore
    if init_div_is_residual is not False:
        with torch.no_grad():
            w2.div_(div_is_residual)  # type: ignore


def _megablocks_sparse_glu_generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: bool = False,
    div_is_residual: float = 1.0,
):
    """Initializes MegaBlocks Sparse GLU.

    Extends the Megablocks Sparse MLP case to an additional weight v1 for GLUs.
    This additional weight v1 has the same initialization procedure as w1 for MLPs.

    Args:
        module (nn.Module): The module to initialize.
        init_fn_ (Callable): Initialization method.
        init_div_is_residual (bool): Flag enabling parameters tagged with _is_residual
            flag to be divided by div_is_residual.
        div_is_residual (float): The value by which parameter initialization is divided
            if init_div_is_residual flag is enabled.
    """
    # Init for w1 and w2 matrices
    _megablocks_sparse_mlp_generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        init_div_is_residual=init_div_is_residual,
        div_is_residual=div_is_residual,
    )

    # Init ported from _megablocks_sparse_mlp_generic_param_init_fn_ for v1
    expert_process_group_size, rank = 1, 0
    if module.expert_parallel_group is not None:
        expert_process_group_size = int(
            module.expert_parallel_group.size(),  # type: ignore
        )
        rank = int(module.expert_parallel_group.rank())  # type: ignore

    hidden_size = int(module.hidden_size)  # type: ignore

    # Separately initialize v1
    v1 = module.v1
    if isinstance(v1, DTensor):
        v1 = v1._local_tensor
    v1_size = list(v1.shape)  # type: ignore
    v1_size[0] = v1_size[0] * expert_process_group_size

    n_exp = v1_size[0] // hidden_size
    _fused = (0, [(n + 1) * hidden_size for n in range(n_exp - 1)])

    _v1 = v1.new_empty(v1_size)  # type: ignore
    fused_param_init_helper(_v1, init_fn_, _fused)
    _v1_local = _v1.chunk(expert_process_group_size, dim=0)[rank]
    with torch.no_grad():
        v1.copy_(_v1_local)  # type: ignore


def _megablocks_mlp_generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: bool = False,
    div_is_residual: float = 1.0,
):
    """Initializes MegaBlocks' MLP.

    To enable elastic deterministic initialization, this method creates the entire
    weight matrix then slice into the weight tensors such that the sampled weights
    should not vary between moe world size for the same random seed.

    Args:
        module (nn.Module): The module to initialize.
        init_fn_ (Callable): Initialization method.
        init_div_is_residual (bool): Flag enabling parameters tagged with _is_residual
            flag to be divided by div_is_residual.
        div_is_residual (float): The value by which parameter initialization is divided
            if init_div_is_residual flag is enabled.
    """
    expert_process_group_size, rank = 1, 0
    if module.expert_parallel_group is not None:
        expert_process_group_size = int(
            module.expert_parallel_group.size(),  # type: ignore
        )
        rank = int(module.expert_parallel_group.rank())  # type: ignore

    _init_fn_ = _flip_fan_mode(init_fn_)

    # Initialize w1
    w1_size = list(module.w1.shape)  # type: ignore
    w1_size[0] = w1_size[0] * expert_process_group_size
    _w1 = module.w1.new_empty(w1_size)  # type: ignore
    stacked_param_init_helper(_w1, _init_fn_, module._stack_dim)  # type: ignore
    _w1_local = _w1.chunk(expert_process_group_size, dim=0)[rank]
    with torch.no_grad():
        module.w1.copy_(_w1_local)  # type: ignore

    # Initialize w2
    w2_size = list(module.w2.shape)  # type: ignore
    w2_size[0] = w2_size[0] * expert_process_group_size
    _w2 = module.w2.new_empty(w2_size)  # type: ignore
    stacked_param_init_helper(_w2, _init_fn_, module._stack_dim)  # type: ignore
    _w2_local = _w2.chunk(expert_process_group_size, dim=0)[rank]
    with torch.no_grad():
        module.w2.copy_(_w2_local)  # type: ignore
    if init_div_is_residual is not False:
        with torch.no_grad():
            module.w2.div_(div_is_residual)  # type: ignore


def _normal_init_(std: float, mean: float = 0.0):
    return partial(torch.nn.init.normal_, mean=mean, std=std)


def _normal_param_init_fn_(
    module: nn.Module,
    std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    init_fn_ = _normal_init_(std=std)

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def baseline_param_init_fn_(
    module: nn.Module,
    init_std: Optional[float],
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    if init_std is None:
        raise ValueError(
            "You must set model.init_config['init_std'] to a float value to use the default initialization scheme.",
        )
    _normal_param_init_fn_(
        module=module,
        std=init_std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def small_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    # Very close to kaiming normal
    # From Transformers without Tears (2019) - Nguyen & Salazar
    std = math.sqrt(2 / (5 * d_model))
    _normal_param_init_fn_(
        module=module,
        std=std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def neox_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022) see
    https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
     and
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs  # unused, just to capture any extra args from the config
    residual_div = n_layers / math.sqrt(10)  # small std / wang std

    small_param_init_fn_(
        module=module,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=residual_div,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def kaiming_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = 'fan_in',
    init_nonlinearity: str = 'leaky_relu',
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

    kaiming_uniform_ = partial(
        nn.init.kaiming_uniform_,
        a=init_gain,
        mode=fan_mode,
        nonlinearity=init_nonlinearity,
    )

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def kaiming_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = 'fan_in',
    init_nonlinearity: str = 'leaky_relu',
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

    kaiming_normal_ = partial(
        torch.nn.init.kaiming_normal_,
        a=init_gain,
        mode=fan_mode,
        nonlinearity=init_nonlinearity,
    )

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def xavier_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    init_gain: float = 0,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def xavier_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[tuple[float, float], float]] = None,
    init_gain: float = 0,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )

from llmfoundry.models.layers.native_sparse_attention import SparseAttention
@summon_dtensor
@module_init_fns.register("sparse_attention")
def sparse_attention_init(
    module: nn.Module,
    init_fn_: Callable,
    d_model: Optional[int],
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs,
) -> bool:
    """
    Initialiser for native SparseAttention.

    We do *not* touch the module’s own parameters, because every learnable tensor
    inside SparseAttention is an nn.Linear or nn.Parameter that will already be
    handled by the 'fc' or default initialisers.  We just need to return True so
    generic_param_init_fn_ knows the module is covered.
    """
    if isinstance(module, SparseAttention):
        return True          # <- claims the module, nothing more to do
    return False

from rotary_embedding_torch import RotaryEmbedding
@module_init_fns.register("rotary_embedding")
@summon_dtensor
def rotary_embedding_init(
    module: nn.Module,
    init_fn_: Callable,
    d_model: Optional[int],
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs,
) -> bool:
    """
    RotaryEmbedding already allocates its cosine/sine Parameter tensors with the
    *final* deterministic values in its own __init__, so we do **not** apply any
    random weight initialisation here.  We only claim the module so that
    `generic_param_init_fn_` stops searching.
    """
    # local import guarantees we compare against the *same* class object

    if isinstance(module, RotaryEmbedding):
        return True          # → parameters considered initialised
    return False


param_init_fns.register('default_', func=torch_default_param_init_fn_)
param_init_fns.register('baseline_', func=baseline_param_init_fn_)
param_init_fns.register('kaiming_uniform_', func=kaiming_uniform_param_init_fn_)
param_init_fns.register('kaiming_normal_', func=kaiming_normal_param_init_fn_)
param_init_fns.register('neox_init_', func=neox_param_init_fn_)
param_init_fns.register('small_init_', func=small_param_init_fn_)
param_init_fns.register('xavier_uniform_', func=xavier_uniform_param_init_fn_)
param_init_fns.register('xavier_normal_', func=xavier_normal_param_init_fn_)

module_init_fns.register('fc', func=fc_init)
module_init_fns.register('embedding', func=embedding_init)
module_init_fns.register('norm', func=norm_init)
module_init_fns.register('multihead_attention', func=multihead_attention_init)
module_init_fns.register('sparse_attention', func=sparse_attention_init)
module_init_fns.register('rotary_embedding', func=rotary_embedding_init)
module_init_fns.register('te_layernorm_mlp', func=te_layernorm_mlp_init)
module_init_fns.register('moe', func=moe_init)
