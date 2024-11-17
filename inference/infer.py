import argparse
import os
import numpy as np
import torch
import tensorflow as tf
from utils import preprocess_input, postprocess_output

def load_model(model_path, framework):
    """
    Load a Neuron-optimized model.
    """
    if framework == "tensorflow":
        print("Loading TensorFlow Neuron model...")
        return tf.saved_model.load(model_path)
    elif framework == "pytorch":
        print("Loading PyTorch Neuron model...")
        return torch.jit.load(model_path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def run_inference(model, input_data, framework):
    """
    Run inference using the loaded model.
    """
    if framework == "tensorflow":
        # TensorFlow: Perform inference using the saved signature
        infer = model.signatures["serving_default"]
        return infer(input_data)
    elif framework == "pytorch":
        # PyTorch: Direct forward pass
        return model(input_data)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on Neuron-optimized models")
    parser.add_argument("--framework", choices=["tensorflow", "pytorch"], required=True, help="Framework of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Neuron-optimized model")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input data (e.g., JSON, CSV, or raw data)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save inference results")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    # Load input data
    input_data = preprocess_input(args.input_data)

    # Load model
    model = load_model(args.model_path, args.framework)

    # Run inference
    print("Running inference...")
    predictions = run_inference(model, input_data, args.framework)

    # Postprocess and save results
    postprocessed_results = postprocess_output(predictions)
    with open(args.output_path, "w") as f:
        f.write(postprocessed_results)

    print(f"Inference results saved to {args.output_path}")

