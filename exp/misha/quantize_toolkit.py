import time
import copy
from typing import Callable, Any, Optional, Tuple
import torch
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def set_quant_backend(engine: str = "fbgemm"):
    assert engine in {"fbgemm", "qnnpack"}
    torch.backends.quantized.engine = engine

def dynamic_int8_quantize(
    model: nn.Module,
    modules: Tuple[type, ...] = (nn.Linear, nn.LSTM, nn.GRU),
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        model, qconfig_spec=modules, dtype=dtype, inplace=False
    )

def static_fx_int8_quantize(
    model: nn.Module,
    example_inputs: Any,
    calibrate_fn: Callable[[nn.Module, Any], None],
    backend: str = "fbgemm",
    custom_mapping: Optional[QConfigMapping] = None,
) -> nn.Module:
    set_quant_backend(backend)
    model = copy.deepcopy(model).eval()
    if custom_mapping is None:
        qconfig = get_default_qconfig(backend)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
    else:
        qconfig_mapping = custom_mapping
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    calibrate_fn(prepared, example_inputs)
    quantized = convert_fx(prepared)
    return quantized

def benchmark_latency(
    model: nn.Module,
    example_inputs: Any,
    inference_fn: Callable[[nn.Module, Any], Any],
    warmup: int = 2,
    iters: int = 5,
) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = inference_fn(model, example_inputs)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = inference_fn(model, example_inputs)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters

def save_model(model: nn.Module, path: str):
    try:
        torch.save(model, path)
    except Exception:
        torch.save(model.state_dict(), path + ".state_dict.pt")