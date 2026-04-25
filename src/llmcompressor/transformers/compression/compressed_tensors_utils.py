import json
import os
import re
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import wraps

import torch
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import (
    ModelCompressor,
    SparsityCompressionConfig,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.offload import from_accelerate, is_rank0, to_accelerate
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.lifecycle.apply import match_targets
from compressed_tensors.utils.match import match_name
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from llmcompressor.core import active_session
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.pytorch.model_load.helpers import copy_python_files_from_model_cache
from llmcompressor.transformers.compression.sparsity_metadata_config import (
    SparsityConfigMetadata,
)
from llmcompressor.transformers.utils import RECIPE_FILE_NAME
from llmcompressor.transformers.utils.helpers import infer_recipe_from_model_path

__all__ = ["modify_save_pretrained"]


_DEFAULT_FUSED_MAPPINGS = [
    [
        r"re:.*(attn|attention)\.q_proj\.weight$",
        r"re:.*(attn|attention)\.k_proj\.weight$",
        r"re:.*(attn|attention)\.v_proj\.weight$",
    ],
    [
        r"re:.*(attn|attention)\.wq_a\.weight$",
        r"re:.*(attn|attention)\.wkv_a_with_mqa\.weight$",
    ],
    [r"re:.*mlp\.gate_proj\.weight$", r"re:.*mlp\.up_proj\.weight$"],
    [r"re:.*w1\.weight$", r"re:.*w3\.weight$"],
]


def modify_save_pretrained(model: PreTrainedModel):
    """
    Overrides a PreTrainedModel's save_pretrained() method with a wrapped version that
    supports compression. The new save_pretrained function performs the following saving
    operations:

    1. Saves the model state, potentially in a compressed format
    2. Saves the recipe, appending any current recipes to existing recipe files
    3. Copies any necessary python files from the model cache
    """

    def save_pretrained_compressed(save_pretrained_method):
        if getattr(save_pretrained_method, "_overridden", False):
            # `model.save_pretrained` has already been replaced, return.
            return save_pretrained_method

        # Keep a weak reference to the model class and unbound save_pretrained
        # method so we can call the original
        model_ref = weakref.ref(save_pretrained_method.__self__)
        original_save_pretrained = save_pretrained_method.__func__
        model_class = model_ref().__class__
        del save_pretrained_method

        @wraps(original_save_pretrained)
        def save_pretrained_wrapper(
            save_directory: str,
            sparsity_config: SparsityCompressionConfig | None = None,
            quantization_format: str | None = None,
            save_compressed: bool = True,
            safe_serialization: bool = True,
            skip_sparsity_compression_stats: bool = True,
            disable_sparse_compression: bool = False,
            **kwargs,
        ):
            """
            Wrapper around PreTrainedModel.save_pretrained(), adds functionality for
            saving models in a compressed format on disk. The compression format is
            saved to the model's config file

            :param save_directory: output directory to save model to
            :param sparsity_config: optional sparsity config to compress model with,
                if no config is provided it will be inferred from the model
            :param quantization_format: optional compression format for quantized
                models. If none is provided it will be inferred from the model
            :param save_compressed: whether or not to compress the model on disk
            :param skip_sparsity_compression_stats: whether to skip the calculation of
                sparsity statistics (such as global sparsity and sparsity structure)
                when saving a model in dense format
            :param disable_sparse_compression: whether to skip sparse compression
                during save, default is False
            :param kwargs: additional kwargs to pass on to model.save_pretrained
            """

            # compress model using compressor
            compressor = get_model_compressor(
                model=model,
                sparsity_config=sparsity_config,
                quantization_format=quantization_format,
                save_compressed=save_compressed,
                skip_sparsity_compression_stats=skip_sparsity_compression_stats,
                disable_sparse_compression=disable_sparse_compression,
            )
            if compressor is not None:
                compressor.compress_model(model)

            # convert to accelerate offloaded for optimal saving with transformers
            to_accelerate(model)

            if is_rank0():
                # save (compressed) model structure
                original_save_pretrained.__get__(model, model_class)(
                    save_directory,
                    safe_serialization=safe_serialization,
                    **kwargs,
                )

                # update config to reflect compression
                if compressor is not None:
                    compressor.update_config(save_directory)

                # ensure regex-matched ignore patterns are in config.json
                _update_config_expanded_ignore(model, save_directory)

                # update existing recipe
                update_and_save_recipe(model.name_or_path, save_directory)

                # copy python files from cache dir to save_path if any
                copy_python_files_from_model_cache(model, save_directory)

                # graft any extra weights (e.g. MTP) from the source checkpoint
                # that were dropped by transformers during from_pretrained
                _graft_extra_weights(model, save_directory, compressor=compressor)

            # convert back from accelerate to restore model to original form
            from_accelerate(model)

        save_pretrained_wrapper._overridden = True
        return save_pretrained_wrapper

    # wrap save_pretrained if not already
    if not getattr(model.save_pretrained, "_overridden", False):
        model.save_pretrained = save_pretrained_compressed(model.save_pretrained)


def get_model_compressor(
    model: torch.nn.Module,
    sparsity_config: SparsityCompressionConfig | None = None,
    quantization_format: str | None = None,
    save_compressed: bool = True,
    skip_sparsity_compression_stats: bool = True,
    disable_sparse_compression: bool = False,
):
    """
    Obtain the compressor based on the config and the
        quantization_format

    :param model: torch model
    :param sparsify_config: Sparsity Compression config
    :param quantization_format: Format that the model was quantized to.
        if not provivided, will be extrapolated from `infer_quantization_format`
    :param save_compressed: boolean representing to save in a compressed
        format
    :param skip_sparsity_compression_stats: bool allowing compression stats on std out
    :param disable_sparse_compression: bool to skip sparse compression
    """

    if sparsity_config is None:
        """
        Case 1: No sparsity config is provided
            1. Will either skip sparsity compression
            2. Or we will infer sparsity from the model directly

        Check recipe for applied sparsity:
            - Set skip_sparsity_compression_stats to False if don't find a
                sparsity structure from the recipe
            - If we identify sparsity based on the recipe or the user
                set skip_sparsity_compression_stats to False, generate config
        """
        sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(
            model, check_only_modifiers=True
        )
        if sparsity_structure is not None:
            skip_sparsity_compression_stats = False

        if skip_sparsity_compression_stats:
            logger.info(
                "skip_sparsity_compression_stats set to True. Skipping sparsity "
                "compression statistic calculations. No sparsity compressor will "
                "be applied."
            )
            sparsity_config = None
        else:
            state_dict = get_state_dict_offloaded_model(model)

            sparsity_config = SparsityConfigMetadata.from_pretrained(
                model,
                state_dict=state_dict,
                compress=save_compressed,
                quantization_format=quantization_format,
                disable_sparse_compression=disable_sparse_compression,
                sparsity_structure=sparsity_structure,
            )
    else:
        """
        # Case 2: User provides a Sparsity Config
            - This is the case when there is existing sparsity in the
                model that we'd like to account for while compressing
            - Users should provide a SparsityConfig, conveying the model's
                sparsity structure when saving the model
        """
        if sparsity_config.sparsity_structure is None:
            logger.info(
                "SparsityConfigMetadata provided without indicating ",
                "the sparsity structure. Sparisty will be inferred from the model. "
                "Consider providing the structure to skip this step ",
            )
            sparsity_config.sparsity_structure = (
                SparsityConfigMetadata.infer_sparsity_structure(model)
            )

    if not save_compressed:
        if quantization_format not in (None, CompressionFormat.dense.value):
            raise ValueError(
                "A quantizatiom format was provided but "
                "save_compressed is set to False. "
                "A compression format can only be applied when "
                "saving the model compressed"
            )
        quantization_format = CompressionFormat.dense.value

    return ModelCompressor.from_pretrained_model(
        model,
        sparsity_config_or_format=sparsity_config,
        quantization_format=quantization_format,
    )


def _update_config_expanded_ignore(
    model: torch.nn.Module, save_directory: str
) -> None:
    """
    Ensure that modules matched by regex ignore patterns are explicitly listed
    in config.json's quantization_config.ignore.

    QuantizationConfig.from_pretrained() builds the ignore list by module type,
    which can miss non-standard module types matched by regex patterns (e.g. MoE
    router modules that are not nn.Linear). The QuantizationModifier stores the
    expanded names on model._quantization_expanded_ignore during on_finalize.
    """
    expanded_ignore = getattr(model, "_quantization_expanded_ignore", None)
    if not expanded_ignore:
        return

    config_path = os.path.join(save_directory, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config_data = json.load(f)

    quant_config = config_data.get("quantization_config")
    if quant_config is None:
        return

    ignore_list = quant_config.get("ignore", [])
    existing = set(ignore_list)

    added = sorted(name for name in expanded_ignore if name not in existing)
    if not added:
        return

    ignore_list.extend(added)
    quant_config["ignore"] = ignore_list

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, sort_keys=True)

    logger.info(
        f"Added {len(added)} regex-matched module(s) to config.json ignore list"
    )


def _is_quantizable_extra_tensor(name: str, tensor: torch.Tensor) -> bool:
    if "." not in name:
        return False

    module_name, param_name = name.rsplit(".", 1)
    return param_name == "weight" and tensor.ndim == 2 and not module_name.endswith("norm")


def _is_tensor_group_scheme(scheme: QuantizationScheme) -> bool:
    if scheme.weights is None:
        return False

    strategy = scheme.weights.strategy
    return strategy in (
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.TENSOR_GROUP.value,
    )


def _resolve_extra_tensor_schemes(
    extra_tensors: dict[str, torch.Tensor],
    quantization_config: QuantizationConfig,
) -> dict[str, QuantizationScheme]:
    target_to_scheme: OrderedDict[str, QuantizationScheme] = OrderedDict()
    for scheme in quantization_config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    if not target_to_scheme:
        return {}

    ignore = quantization_config.ignore or []
    tensor_schemes: dict[str, QuantizationScheme] = {}
    for tensor_name, tensor in extra_tensors.items():
        if not _is_quantizable_extra_tensor(tensor_name, tensor):
            continue

        module_name, _ = tensor_name.rsplit(".", 1)
        module = torch.nn.Linear(
            tensor.shape[1],
            tensor.shape[0],
            bias=False,
            dtype=tensor.dtype,
        )
        if match_targets(module_name, module, ignore):
            continue

        matched_targets = match_targets(module_name, module, target_to_scheme.keys())
        if not matched_targets:
            continue

        scheme = target_to_scheme[matched_targets[0]]
        if scheme.weights is None:
            continue

        tensor_schemes[tensor_name] = scheme

    return tensor_schemes


def _initialize_extra_quantized_linear(
    tensor: torch.Tensor, scheme: QuantizationScheme
) -> torch.nn.Linear:
    out_features, in_features = tensor.shape
    module = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    module.weight.data.copy_(tensor)
    initialize_module_for_quantization(
        module,
        deepcopy(scheme),
        force_zero_point=False,
    )
    return module


def _calibrate_extra_weight_global_scale(module: torch.nn.Linear) -> None:
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    update_weight_global_scale(module)
    freeze_module_quantization(module)


def _calibrate_extra_weight_scale(module: torch.nn.Linear) -> None:
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    update_weight_zp_scale(module)
    freeze_module_quantization(module)


def _match_names_set_eager(
    names: list[str],
    targets: list[str],
) -> tuple[list[dict[str, str | None]], dict[str, str | None] | None]:
    matched_sets = []
    matches = dict.fromkeys(targets, None)

    def natural_key(name: str) -> list[str | int]:
        return [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
        ]

    for name in sorted(names, key=natural_key):
        for target in targets:
            if not match_name(name, target):
                continue

            if matches[target] is None:
                matches[target] = name
            else:
                raise ValueError(
                    f"Matched a {target} twice before completing set "
                    f"({matches[target]}, {name})"
                )

        if all(matches[target] is not None for target in targets):
            matched_sets.append(matches)
            matches = dict.fromkeys(targets, None)

    unmatched = matches if any(value is not None for value in matches.values()) else None
    return matched_sets, unmatched


def _get_extra_fused_names(
    tensor_names: list[str],
) -> tuple[list[dict[str, str | None]], list[dict[str, str | None]]]:
    matched_sets = []
    unmatched_sets = []
    for mapping in _DEFAULT_FUSED_MAPPINGS:
        matched, unmatched = _match_names_set_eager(tensor_names, mapping)
        matched_sets.extend(matched)
        if unmatched is not None:
            unmatched_sets.append(unmatched)

    return matched_sets, unmatched_sets


def _replace_extra_weight_with_quantized_state(
    extra_tensors: dict[str, torch.Tensor],
    tensor_name: str,
    module: torch.nn.Linear,
) -> None:
    del extra_tensors[tensor_name]

    module_name, _ = tensor_name.rsplit(".", 1)
    for key, value in module.state_dict(prefix=f"{module_name}.").items():
        extra_tensors[key] = value.to("cpu")


def _compress_extra_quantized_module(
    module: torch.nn.Linear,
    compressor: ModelCompressor,
) -> None:
    container = torch.nn.Module()
    container.add_module("layer", module)
    compressor.compress_model(container)

    # `compressed_tensors` has shipped this cleanup both as a helper method and
    # as a direct `ct_decompress_hook` attribute on the model.
    if hasattr(compressor, "remove_decompression_hook"):
        compressor.remove_decompression_hook(container)
        return

    hook = getattr(container, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()
        delattr(container, "ct_decompress_hook")


def _quantize_extra_tensors(
    extra_tensors: dict[str, torch.Tensor],
    compressor: ModelCompressor | None,
) -> dict[str, torch.Tensor]:
    if compressor is None or compressor.quantization_config is None:
        return extra_tensors

    tensor_schemes = _resolve_extra_tensor_schemes(
        extra_tensors,
        compressor.quantization_config,
    )
    if not tensor_schemes:
        return extra_tensors

    quantized_tensors = dict(extra_tensors)
    processed: set[str] = set()

    tensor_group_names = [
        tensor_name
        for tensor_name, scheme in tensor_schemes.items()
        if _is_tensor_group_scheme(scheme)
    ]
    fused_groups: dict[str, list[str]] = {}
    if tensor_group_names:
        matched_sets, unmatched_sets = _get_extra_fused_names(tensor_group_names)
        for unmatched in unmatched_sets:
            fallback_names = sorted(name for name in unmatched.values() if name is not None)
            if fallback_names:
                logger.warning(
                    "Falling back to per-tensor calibration for {} extra tensor-group weight(s)",
                    len(fallback_names),
                )

        for matched in matched_sets:
            fused_names = [name for name in matched.values() if name is not None]
            if len(fused_names) < 2:
                continue

            first_scheme = tensor_schemes[fused_names[0]]
            if any(tensor_schemes[name] != first_scheme for name in fused_names[1:]):
                continue

            for name in fused_names:
                fused_groups[name] = fused_names

    for tensor_name, scheme in tensor_schemes.items():
        if tensor_name in processed:
            continue

        fused_names = fused_groups.get(tensor_name)
        if fused_names is not None:
            modules = {
                name: _initialize_extra_quantized_linear(quantized_tensors[name], tensor_schemes[name])
                for name in fused_names
            }
            for module in modules.values():
                _calibrate_extra_weight_global_scale(module)

            fused_global_scale = torch.min(
                torch.cat(
                    [module.weight_global_scale.reshape(-1) for module in modules.values()],
                    dim=0,
                )
            )

            for name, module in modules.items():
                module.weight_global_scale.data.copy_(
                    torch.full_like(module.weight_global_scale, fused_global_scale)
                )
                _calibrate_extra_weight_scale(module)
                _compress_extra_quantized_module(module, compressor)
                _replace_extra_weight_with_quantized_state(quantized_tensors, name, module)
                processed.add(name)

            continue

        module = _initialize_extra_quantized_linear(quantized_tensors[tensor_name], scheme)
        if _is_tensor_group_scheme(scheme):
            _calibrate_extra_weight_global_scale(module)

        _calibrate_extra_weight_scale(module)
        _compress_extra_quantized_module(module, compressor)
        _replace_extra_weight_with_quantized_state(quantized_tensors, tensor_name, module)
        processed.add(tensor_name)

    if processed:
        logger.info("Quantized {} extra weight tensor(s) during grafting", len(processed))

    return quantized_tensors


def _graft_extra_weights(
    model: PreTrainedModel,
    save_directory: str,
    compressor: ModelCompressor | None = None,
) -> None:
    """
    Copy any weight keys present in the source checkpoint but missing from the
    output directory.  This handles weights that transformers' from_pretrained()
    silently drops (e.g. MTP / multi-token-prediction weights in Qwen3.5 models
    that are filtered via ``_keys_to_ignore_on_load_unexpected``).

    The function is generic: it discovers *all* keys present in the source but
    absent from the output whose parent module doesn't exist in the model, so it
    works for any model architecture that carries extra modules unknown to
    transformers.  Keys that are merely *renamed* by compression (e.g.
    ``weight`` → ``weight_packed``) are excluded because their parent module
    still exists in the model.

    :param model: the model that was loaded and (possibly) quantized
    :param save_directory: path to the output directory produced by
        ``save_pretrained``
    :param compressor: optional save-time model compressor. If present, extra
        linear weights that match its quantization config are quantized before
        being written to ``extra_weights.safetensors``.
    """
    from transformers.utils.hub import cached_file

    source_name = model.name_or_path

    # ------------------------------------------------------------------
    # 1. Resolve source weight map  (key -> shard filename)
    # ------------------------------------------------------------------
    source_weight_map: dict[str, str] = {}  # key -> shard filename
    try:
        index_path = cached_file(source_name, SAFE_WEIGHTS_INDEX_NAME)
        with open(index_path, "r") as f:
            source_index = json.load(f)
        source_weight_map = source_index["weight_map"]
    except Exception:
        # Single-shard model – enumerate keys directly
        try:
            single_path = cached_file(source_name, SAFE_WEIGHTS_NAME)
            with safe_open(single_path, framework="pt") as f:
                source_weight_map = {k: SAFE_WEIGHTS_NAME for k in f.keys()}
        except Exception:
            logger.debug(
                "Could not resolve source safetensors for '{}'; "
                "skipping extra-weight grafting.",
                source_name,
            )
            return

    if not source_weight_map:
        return

    # ------------------------------------------------------------------
    # 2. Enumerate keys already saved in the output directory
    # ------------------------------------------------------------------
    output_keys: set[str] = set()
    output_index_path = _find_safetensors_index(save_directory)
    if output_index_path is not None:
        with open(output_index_path, "r") as f:
            output_index = json.load(f)
        output_keys = set(output_index["weight_map"].keys())
    else:
        single_output = os.path.join(save_directory, SAFE_WEIGHTS_NAME)
        if os.path.exists(single_output):
            with safe_open(single_output, framework="pt") as f:
                output_keys = set(f.keys())

    if not output_keys:
        # Output has no safetensors (e.g. safe_serialization=False) — skip
        return

    # ------------------------------------------------------------------
    # 3. Compute extra keys — only those whose parent module is absent
    # ------------------------------------------------------------------
    # Keys missing from the output fall into two categories:
    #   a) Truly dropped by transformers (e.g. mtp.* keys) — the parent
    #      module was never created, so it doesn't exist in the model.
    #   b) Renamed by compression (e.g. weight → weight_packed) — the
    #      parent module still exists, just with different parameter names.
    # We only want category (a).
    candidate_keys = set(source_weight_map.keys()) - output_keys
    if not candidate_keys:
        return

    model_module_names = {name for name, _ in model.named_modules()}

    extra_keys = set()
    for key in candidate_keys:
        # Extract parent module path: "a.b.c.weight" → "a.b.c"
        module_path = key.rsplit(".", 1)[0] if "." in key else ""
        if module_path not in model_module_names:
            extra_keys.add(key)

    if not extra_keys:
        return

    logger.info(
        "Grafting {} extra weight key(s) from source checkpoint "
        "(e.g. {})",
        len(extra_keys),
        next(iter(sorted(extra_keys))),
    )

    # ------------------------------------------------------------------
    # 4. Load extra tensors from source (memory-efficient: per-shard)
    # ------------------------------------------------------------------
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in extra_keys:
        keys_by_shard[source_weight_map[key]].append(key)

    extra_tensors: dict[str, torch.Tensor] = {}
    for shard_filename, keys in keys_by_shard.items():
        shard_path = cached_file(source_name, shard_filename)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                extra_tensors[key] = f.get_tensor(key)

    extra_tensors = _quantize_extra_tensors(extra_tensors, compressor)

    # ------------------------------------------------------------------
    # 5. Save extra tensors to a new shard
    # ------------------------------------------------------------------
    extra_shard_name = "extra_weights.safetensors"
    extra_shard_path = os.path.join(save_directory, extra_shard_name)
    save_file(extra_tensors, extra_shard_path)

    extra_size = sum(
        t.nelement() * t.element_size() for t in extra_tensors.values()
    )

    # ------------------------------------------------------------------
    # 6. Update (or create) the safetensors index
    # ------------------------------------------------------------------
    if output_index_path is not None:
        # Multi-shard output – update existing index
        with open(output_index_path, "r") as f:
            index_data = json.load(f)

        for key in sorted(extra_tensors):
            index_data["weight_map"][key] = extra_shard_name

        metadata = index_data.get("metadata", {})
        old_total = metadata.get("total_size", 0)
        metadata["total_size"] = old_total + extra_size
        index_data["metadata"] = metadata

        with open(output_index_path, "w") as f:
            json.dump(index_data, f, indent=2, sort_keys=False)
            f.write("\n")
    else:
        # Single-shard output – create a new index
        weight_map: dict[str, str] = {}
        original_size = 0
        single_output = os.path.join(save_directory, SAFE_WEIGHTS_NAME)
        if os.path.exists(single_output):
            with safe_open(single_output, framework="pt") as f:
                for key in f.keys():
                    weight_map[key] = SAFE_WEIGHTS_NAME
                    t = f.get_tensor(key)
                    original_size += t.nelement() * t.element_size()

        for key in sorted(extra_tensors):
            weight_map[key] = extra_shard_name

        index_data = {
            "metadata": {"total_size": original_size + extra_size},
            "weight_map": weight_map,
        }

        index_path = os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME)
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2, sort_keys=False)
            f.write("\n")

    logger.info(
        "Saved extra weights to {} ({:.1f} MB)",
        extra_shard_name,
        extra_size / (1024 * 1024),
    )


def _find_safetensors_index(directory: str) -> str | None:
    """Return the path to the safetensors index JSON in *directory*, or None."""
    for name in os.listdir(directory):
        if name.endswith("safetensors.index.json"):
            return os.path.join(directory, name)
    return None


def update_and_save_recipe(model_stub: str, save_directory: str):
    """
    Save a recipe ontop of any existing recipe files located at model_stub

    :param model_stub: path to existing model or model stub which may contain an
        existing recipe
    :param save_directory: path to save combined existing recipe and current recipe
    """

    existing_recipe = infer_recipe_from_model_path(model_stub)

    recipe = active_session().lifecycle.recipe

    recipe_path = os.path.join(save_directory, RECIPE_FILE_NAME)
    recipe.yaml(file_path=recipe_path, existing_recipe_path=existing_recipe)
