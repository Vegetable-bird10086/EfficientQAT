from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_hf_token(token: Optional[str] = None, use_auth_token: Optional[bool] = None) -> Optional[Any]:
    if token:
        return token
    if use_auth_token:
        return True
    return None


def _with_common_hf_kwargs(kwargs: Dict[str, Any], trust_remote_code: bool = False, token: Optional[Any] = None) -> Dict[str, Any]:
    output = dict(kwargs)
    if trust_remote_code:
        output["trust_remote_code"] = True
    if token is not None:
        output["token"] = token
    return output


def load_auto_tokenizer(
    model_name_or_path: str,
    *,
    use_fast: bool = False,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    legacy: Optional[bool] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    load_kwargs["use_fast"] = use_fast
    if legacy is not None:
        load_kwargs["legacy"] = legacy
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError as exc:
        if "legacy" in str(exc) and "legacy" in load_kwargs:
            load_kwargs.pop("legacy", None)
            return AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        raise


def load_auto_config(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    return AutoConfig.from_pretrained(model_name_or_path, **load_kwargs)


def load_auto_model_for_causal_lm(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)


def build_model_from_config(
    config,
    *,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    load_kwargs = dict(kwargs)
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    return AutoModelForCausalLM.from_config(config=config, **load_kwargs)


def tokenizer_is_llama_like(tokenizer) -> bool:
    name = tokenizer.__class__.__name__.lower()
    return "llama" in name
