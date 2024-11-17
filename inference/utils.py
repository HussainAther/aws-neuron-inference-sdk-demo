import json
import numpy as np
import torch
import tensorflow as tf

def preprocess_input(input_path):
    """
    Preprocess input data for inference.
    Example: Load JSON or CSV and convert it into a suitable format for the model.
    """
    print("Preprocessing input data...")
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
        # Example: Convert JSON to a NumPy array or Tensor
        input_data = np.array(data["input_data"], dtype=np.float32)
    elif input_path.endswith(".csv"):
        import pandas as pd
        data = pd.read_csv(input_path)
        input_data = data.to_numpy(dtype=np.float32)
    else:
        raise ValueError("Unsupported input file format. Use JSON or CSV.")

    # Convert to TensorFlow Tensor or PyTorch Tensor
    if isinstance(input_data, np.ndarray):
        if "torch" in str(type(tf)):  # TensorFlow
            return tf.convert_to_tensor(input_data)
        else:  # PyTorch
            return torch.tensor(input_data)
    else:
        raise ValueError("Failed to preprocess input.")

def postprocess_output(predictions):
    """
    Postprocess model predictions for output.
    Example: Convert raw predictions to human-readable format (e.g., labels, scores).
    """
    print("Postprocessing predictions...")
    if isinstance(predictions, dict):  # TensorFlow output
        # Example: Extract specific tensors
        return json.dumps({key: val.numpy().tolist() for key, val in predictions.items()})
    elif torch.is_tensor(predictions):  # PyTorch output
        # Example: Convert tensor to JSON-friendly format
        return json.dumps(predictions.cpu().detach().numpy().tolist())
    else:
        raise ValueError("Unsupported predictions format.")

