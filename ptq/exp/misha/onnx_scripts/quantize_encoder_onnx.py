from pathlib import Path
import argparse
from typing import List, Optional, Dict, Any, Iterator
import numpy as np

from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
import onnx


class EncoderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_names: List[str], samples: List[np.ndarray]):
        self.input_names = input_names
        self.samples = samples
        self._iter: Optional[Iterator[np.ndarray]] = None

    def get_next(self) -> Optional[Dict[str, Any]]:
        if self._iter is None:
            self._iter = iter(self.samples)
        try:
            tokens = next(self._iter)
        except StopIteration:
            return None
        lengths = np.array([tokens.shape[1]], dtype=np.int64)  # [1]
        sid = np.array([1], dtype=np.int64)  # single speaker default
        return {
            self.input_names[0]: tokens,   # tokens [1, T] int64
            self.input_names[1]: lengths,  # lengths [1] int64
            self.input_names[2]: sid,      # sid [1] int64
        }


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime QDQ quantization for encoder")
    parser.add_argument("--onnx", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--out", type=str, default="", help="Output int8 onnx path")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--calib_count", type=int, default=64)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out) if args.out else onnx_path.with_suffix(".int8.onnx")

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )
        print(f"Dynamic QDQ quantized encoder saved to: {out_path}")
        return

    # Static calibration
    m = onnx.load(str(onnx_path))
    input_names = [i.name for i in m.graph.input]  # [tokens, lengths, sid]

    # Make some synthetic calibration samples (variable T)
    samples: List[np.ndarray] = []
    rng = np.random.default_rng(0)
    for _ in range(args.calib_count):
        T = int(rng.integers(10, 100))
        tokens = rng.integers(low=1, high=62, size=(1, T), dtype=np.int64)
        samples.append(tokens)

    dr = EncoderCalibrationDataReader(input_names, samples)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Static QDQ quantized encoder saved to: {out_path}")


if __name__ == "__main__":
    main()




