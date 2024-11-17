#!/bin/bash

# Exit script on error
set -e

echo "Setting up AWS Neuron SDK and dependencies for Inferentia..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install basic dependencies
echo "Installing basic dependencies..."
sudo apt-get install -y build-essential curl wget git

# Add AWS Neuron repository and GPG key
echo "Adding AWS Neuron repository..."
curl -fsSL https://apt.repos.neuron.amazonaws.com/ubuntu/dists/focal/main/gpg | sudo gpg --dearmor -o /usr/share/keyrings/aws-neuron-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/aws-neuron-keyring.gpg] https://apt.repos.neuron.amazonaws.com/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/aws-neuron.list

# Update package lists
echo "Updating package lists with Neuron repository..."
sudo apt-get update -y

# Install AWS Neuron SDK
echo "Installing AWS Neuron SDK..."
sudo apt-get install -y aws-neuron-dkms aws-neuron-runtime-base aws-neuron-tools

# Install Python packages for Neuron
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
pip install neuron tensorflow-neuronx torch-neuronx

# Verify installation
echo "Verifying Neuron installation..."
neuron-cli --version
python3 -c "import tensorflow_neuronx; print('TensorFlow NeuronX installed')"
python3 -c "import torch_neuronx; print('PyTorch NeuronX installed')"

echo "AWS Neuron SDK setup is complete."

