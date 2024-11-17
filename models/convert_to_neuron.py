import argparse
import os
import json
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def convert_tensorflow_model(input_path, output_path, config):
    """
    Convert a TensorFlow model to Neuron-optimized format.
    """
    print("Loading TensorFlow model...")
    model = tf.saved_model.load(input_path)

    print("Tracing TensorFlow model with Neuron...")
    # Prepare input signature based on config
    input_signatures = [
        tf.TensorSpec(config["input_shapes"]["input_ids"], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(config["input_shapes"]["attention_mask"], dtype=tf.int32, name="attention_mask")
    ]
    compiled_model = tf.neuron.trace(model, input_signatures)

    print(f"Saving TensorFlow Neuron-optimized model to {output_path}...")
    compiled_model.save(output_path)
    print("Conversion to Neuron completed.")

def convert_pytorch_model(input_path, output_path, config):
    """
    Convert a PyTorch model to Neuron-optimized format.
    """
    print("Loading PyTorch model...")
    model = AutoModelForSequenceClassification.from_pretrained(input_path)
    tokenizer = AutoTokenizer.from_pretrained(input_path)

    # Create dummy inputs based on the configuration
    print("Creating dummy inputs for tracing...")
    dummy_input = tokenizer(
        "This is a sample input for Neuron conversion.",
        return_tensors="pt",
        max_length=config["input_shapes"]["input_ids"][1],  # Use sequence length from config
        padding="max_length",
        truncation=True
    )

    print("Tracing PyTorch model with Neuron...")
    neuron_model = torch.neuron.trace(model, example_inputs=(dummy_input["input_ids"], dummy_input["attention_mask"]))

    print(f"Saving PyTorch Neuron-optimized model to {output_path}...")
    neuron_model.save(output_path)
    print("Conversion to Neuron completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models to Neuron-optimized format")
    parser.add_argument("--framework", choices=["tensorflow", "pytorch"], required=True, help="Framework of the input model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input pretrained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the Neuron-optimized model")
    parser.add_argument("--config", type=str, default="setup/neuron_config.json", help="Path to Neuron configuration file")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input model not found at {args.input_path}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found at {args.config}")

    # Load configuration
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    # Convert the model based on the framework
    if args.framework == "tensorflow":
        convert_tensorflow_model(args.input_path, args.output_path, config)
    elif args.framework == "pytorch":
        convert_pytorch_model(args.input_path, args.output_path, config)
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")

