import netron

# Replace with your ONNX file path
onnx_model_path = "tfpp.onnx"

# Launch Netron UI
netron.start(onnx_model_path, browse=False)
