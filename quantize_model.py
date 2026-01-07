import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

model_input = "models/best.onnx"
model_output = "models/best_int8.onnx"

quantize_dynamic(model_input, model_output, weight_type=QuantType.QUInt8)

print(f"Quantized model saved to {model_output}")
