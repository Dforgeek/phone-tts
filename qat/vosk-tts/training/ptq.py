# quantize_toolkit.py
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Any
import torch
from torch import nn
from torch.ao.quantization import QConfig, HistogramObserver, MinMaxObserver

from torch.ao.quantization import (
    QConfig,
    HistogramObserver,
    MinMaxObserver,
    QuantWrapper,
)
# ---------- Utility wrappers ----------

class QuantizedModuleWrapper(nn.Module):
    """
    Generic wrapper:
      - quantizes *all* tensor inputs (recursively in tuples/lists/dicts)
      - runs the original module
      - dequantizes *all* tensor outputs

    You can wrap dec, dp, flow, enc_p.encoder, etc.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def _apply_to_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor], obj: Any) -> Any:
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, (tuple, list)):
            return type(obj)(self._apply_to_tensors(fn, x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._apply_to_tensors(fn, v) for k, v in obj.items()}
        return obj

    def forward(self, *args, **kwargs):
        # Quantize all tensor inputs
        q_args = [self._apply_to_tensors(self.quant, a) for a in args]
        q_kwargs = {k: self._apply_to_tensors(self.quant, v) for k, v in kwargs.items()}

        out = self.module(*q_args, **q_kwargs)

        # Dequantize all tensor outputs
        out = self._apply_to_tensors(self.dequant, out)
        return out




def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    """
    "dec.ups.0" -> root.dec.ups.0
    """
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m


def _wrap_convs_and_linears_in_module(
    module: nn.Module,
    qconfig: QConfig,
) -> None:
    """
    Рекурсивно оборачиваем ТОЛЬКО conv / convT / linear в QuantWrapper.
    Внешний интерфейс этих модулей остаётся float → остальные операции не видят int8.
    """
    for name, child in list(module.named_children()):
        # если это conv / convT / linear — оборачиваем
        if isinstance(
            child,
            (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear),
        ):
            wrapped = QuantWrapper(child)
            wrapped.qconfig = qconfig
            setattr(module, name, wrapped)
        else:
            # рекурсивно спускаемся дальше
            _wrap_convs_and_linears_in_module(child, qconfig)

# ---------- Public API: PTQ ----------

def _make_per_tensor_qconfig(backend: str) -> QConfig:
    # backend всё равно выставляем, чтобы torch знал, какие опкоды использовать
    torch.backends.quantized.engine = backend

    # Активации — per-tensor affine (quint8)
    activation = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )

    # Веса — per-tensor symmetric (qint8), БЕЗ per-channel
    weight = MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )

    return QConfig(activation=activation, weight=weight)



## ТОЛЬКО ДЛЯ СВЁРТОК
def prepare_model_for_ptq_convs_only(
    model: nn.Module,
    module_roots: Optional[Sequence[str]] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Подготовка к PTQ, но:
      - НЕ трогаем большие блоки целиком (dec, flow, encoder),
      - оборачиваем ТОЛЬКО conv/convT/linear под указанными корнями.

    module_roots:
      - None  -> искать conv/linear по ВСЕЙ модели
      - ["dec"] -> искать только внутри model.dec
      - ["dec", "enc_q"] -> только декодер + постериор-энкодер и т.д.
    """
    qconfig = _make_per_tensor_qconfig(backend)

    if module_roots is None or len(module_roots) == 0:
        _wrap_convs_and_linears_in_module(model, qconfig)
    else:
        for root_name in module_roots:
            sub = _get_submodule(model, root_name)
            _wrap_convs_and_linears_in_module(sub, qconfig)

    # на корневую модель qconfig НЕ вешаем,
    # чтобы PyTorch не пытался квантовать всё подряд
    model.qconfig = None

    torch.ao.quantization.prepare(model, inplace=True)
    return model


def calibrate_model(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
) -> None:
    """
    Run user-provided calibration_fn with observers active.
    calibration_fn should:
      - put model in eval() mode
      - run several inference-like forward passes
      - not change model structure
    """
    model.eval()
    with torch.inference_mode():
        calibration_fn(model)


def convert_model_from_ptq(model: nn.Module) -> nn.Module:
    """
    Replace float modules + observers with quantized modules.
    """
    torch.ao.quantization.convert(model, inplace=True)
    return model



### квантизуем ТОЛЬКО СВЁРТКИ И ЛИНЕЙНЫЕ
def quantize_ptq_convs_only(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    module_roots: Optional[Sequence[str]] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    End-to-end PTQ, но:
      - готовим только conv/convT/linear (через QuantWrapper),
      - всё остальное остаётся float.

    module_roots:
      - см. prepare_model_for_ptq_convs_only
    """
    prepare_model_for_ptq_convs_only(
        model,
        module_roots=module_roots,
        backend=backend,
    )
    calibrate_model(model, calibration_fn)
    convert_model_from_ptq(model)
    return model

# ---------- Public API: QAT-ready (no training loop here) ----------

# def prepare_model_for_qat(
#     model: nn.Module,
#     module_names: Sequence[str],
#     backend: str = "fbgemm",
# ) -> nn.Module:
#     """
#     Same wrapping as PTQ, but uses QAT qconfig and prepare_qat.
#     After this, you can fine-tune the model for a few epochs.
#     """
#     assert len(module_names) > 0, "module_names must not be empty"

#     torch.backends.quantized.engine = backend
#     qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)

#     _wrap_convs_and_linears_in_module(model, module_names, qconfig)
#     model.qconfig = None

#     torch.ao.quantization.prepare_qat(model, inplace=True)
#     return model


# def finish_qat_and_convert(model: nn.Module) -> nn.Module:
#     """
#     After QAT training is done (fake-quant), convert to real quantized modules.
#     """
#     model.eval()
#     torch.ao.quantization.convert(model, inplace=True)
#     return model


from typing import Optional, Sequence
import torch
from torch import nn

# ... остальной код как у тебя выше ...
from torch.ao.quantization import qconfig as tq_config


def prepare_model_for_qat_convs_only(
    model: nn.Module,
    module_roots: Optional[Sequence[str]] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    QAT-эквивалент prepare_model_for_ptq_convs_only:
      - оборачиваем ТОЛЬКО conv/convT/linear в QuantWrapper
      - вешаем на них QAT qconfig
      - вызываем prepare_qat (fake-quant вставится внутрь QuantWrapper-ов)
    """
    torch.backends.quantized.engine = backend
    qconfig = tq_config.default_qat_qconfig

    if module_roots is None or len(module_roots) == 0:
        # по всей модели
        _wrap_convs_and_linears_in_module(model, qconfig)
    else:
        # только внутри указанных сабмодулей, например ["dec"]
        for root_name in module_roots:
            sub = _get_submodule(model, root_name)
            _wrap_convs_and_linears_in_module(sub, qconfig)

    # ВАЖНО: на корень qconfig = None, чтобы не квантизовать всё подряд
    model.qconfig = None

    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def finish_qat_and_convert(model: nn.Module) -> nn.Module:
    """
    После дообучения (QAT) конвертим fake-quant в реальные квантованные модули.
    """
    model.eval()
    torch.ao.quantization.convert(model, inplace=True)
    return model
