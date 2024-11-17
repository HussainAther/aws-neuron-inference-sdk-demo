# AWS Neuron Inference SDK Demo

This repository demonstrates how to convert and run machine learning models optimized for AWS Inferentia-based instances using the AWS Neuron SDK and Inference SDK.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Directory Structure](#directory-structure)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction

AWS Neuron is a software development kit (SDK) that optimizes and deploys machine learning models on AWS Inferentia hardware. This repository provides a complete pipeline for:
- Converting TensorFlow and PyTorch models into Neuron-optimized models.
- Running inference with optimized models.
- Profiling and debugging Neuron models.

---

## Features

- **Framework Support**: Compatible with TensorFlow and PyTorch.
- **Model Conversion**: Automatically convert pretrained models to Neuron-optimized formats.
- **Inference Pipeline**: Run inference efficiently and log performance metrics.
- **Configurable**: Customize batch sizes, input shapes, precision, and debugging options using `neuron_config.json`.

---

## Setup

### Prerequisites
1. AWS Inferentia-based EC2 instance (e.g., Inf1).
2. Python 3.8 or later.
3. Ensure `pip` is installed and up-to-date:
   ```bash
   python3 -m pip install --upgrade pip
   ```

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/aws-neuron-inference-sdk-demo.git
   cd aws-neuron-inference-sdk-demo
   ```
2. Run the setup script to install dependencies:
   ```bash
   bash setup/setup_inferentia.sh
   ```

---

## Usage

### Convert Pretrained Models
Use the `convert_to_neuron.py` script to convert a model into Neuron format.

- TensorFlow example:
  ```bash
  python convert_to_neuron.py --framework tensorflow \
      --input_path pretrained_model/my_tf_model \
      --output_path neuron_models/my_tf_neuron_model
  ```
- PyTorch example:
  ```bash
  python convert_to_neuron.py --framework pytorch \
      --input_path pretrained_model/my_torch_model \
      --output_path neuron_models/my_torch_neuron_model
  ```

### Run Inference
Use the `infer.py` script to perform inference with a Neuron-optimized model.

- TensorFlow example:
  ```bash
  python infer.py --framework tensorflow \
      --model_path neuron_models/my_tf_neuron_model \
      --input_data input_data.json \
      --output_path results.json
  ```
- PyTorch example:
  ```bash
  python infer.py --framework pytorch \
      --model_path neuron_models/my_torch_model.pt \
      --input_data input_data.json \
      --output_path results.json
  ```

### Configuration
Modify the `setup/neuron_config.json` file to adjust input shapes, batch sizes, precision, and more.

---

## Directory Structure

```plaintext
aws-neuron-inference-sdk-demo/
│
├── README.md                  # Overview and usage instructions
├── .gitignore                 # Ignore unnecessary files
├── requirements.txt           # Python dependencies
├── setup/
│   ├── setup_inferentia.sh    # Script to set up Inferentia and Neuron SDK
│   └── neuron_config.json     # Example Neuron configuration
│
├── pretrained_model/          # Directory to store pretrained models
│   └── README.md              # Instructions for adding models
│
├── models/
│   ├── convert_to_neuron.py   # Convert models to Neuron-optimized format
│
├── inference/
│   ├── infer.py               # Script to run inference using Neuron SDK
│   └── utils.py               # Helper functions
│
├── tests/
│   ├── test_conversion.py     # Test model conversion pipeline
│   └── test_inference.py      # Test inference accuracy and latency
│
└── docs/
    ├── setup_guide.md         # Detailed setup instructions
    └── troubleshooting.md     # Common issues and fixes
```

---

## Examples

### Input Data
Prepare your input data in JSON format:
```json
{
  "input_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}
```

### Inference Results
The output is saved in JSON format. Example:
```json
[
  {"label": "positive", "score": 0.98},
  {"label": "negative", "score": 0.45}
]
```

---

## Troubleshooting

- **Error: Model conversion failed**:
  - Verify the input model is compatible with Neuron (check AWS documentation).
  - Ensure `neuron_config.json` has the correct input shapes.
- **Error: Neuron runtime not found**:
  - Ensure you are running on an Inferentia-based instance and the Neuron SDK is installed.

For more details, check the [troubleshooting guide](docs/troubleshooting.md).

---

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes and submit a pull request.

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

