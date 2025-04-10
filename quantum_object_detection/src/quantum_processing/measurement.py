"""
measurement.py - Quantum measurement module for hybrid quantum-classical object detection.

This module handles measurement of quantum circuits and conversion of quantum features
to classical format suitable for input to YOLO model.
"""

import numpy as np
import qiskit
from typing import List, Dict, Union, Tuple, Optional
from qiskit import QuantumCircuit, transpile, Aer, execute


def measure_quantum_state(circuit: QuantumCircuit, shots: int = 1024) -> Dict:
    """
    Measure the quantum state of the circuit and return measurement results.
    
    Args:
        circuit: The quantum circuit to measure
        shots: Number of shots for measurement (default: 1024)
        
    Returns:
        Dictionary containing measurement outcomes and counts
    """
    # Add measurement to all qubits if not already present
    meas_circ = circuit.copy()
    if not meas_circ.data or not any(instr.operation.name == 'measure' for instr in meas_circ.data):
        meas_qubits = list(range(meas_circ.num_qubits))
        meas_circ.measure(meas_qubits, meas_qubits)
    
    # Execute the circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    compiled_circuit = transpile(meas_circ, simulator)
    result = execute(compiled_circuit, simulator, shots=shots).result()
    
    # Get counts and return
    counts = result.get_counts(compiled_circuit)
    return counts


def extract_features_from_measurements(
    measurement_results: Dict, 
    feature_dim: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Convert quantum measurement results to classical feature vectors.
    
    Args:
        measurement_results: Dictionary of measurement outcomes and counts
        feature_dim: Optional tuple (height, width) for reshaping features
        
    Returns:
        Numpy array containing extracted features
    """
    # Convert measurement outcomes to probabilities
    total_shots = sum(measurement_results.values())
    probabilities = {k: v / total_shots for k, v in measurement_results.items()}
    
    # Convert probabilities to feature vector
    # Sort to ensure consistent ordering
    sorted_bitstrings = sorted(probabilities.keys())
    feature_vector = np.array([probabilities[k] for k in sorted_bitstrings])
    
    # Reshape if dimensions are provided
    if feature_dim is not None:
        # Ensure the feature vector length matches the product of dimensions
        expected_length = feature_dim[0] * feature_dim[1]
        current_length = len(feature_vector)
        
        if current_length < expected_length:
            # Pad with zeros if necessary
            feature_vector = np.pad(feature_vector, (0, expected_length - current_length))
        elif current_length > expected_length:
            # Truncate if necessary
            feature_vector = feature_vector[:expected_length]
        
        feature_vector = feature_vector.reshape(feature_dim)
    
    return feature_vector


def process_multiple_circuits(
    circuits: List[QuantumCircuit],
    shots: int = 1024,
    feature_dim: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Process multiple quantum circuits and combine their measurement results.
    
    Args:
        circuits: List of quantum circuits to measure
        shots: Number of shots for each measurement
        feature_dim: Optional tuple (height, width) for reshaping each feature map
        
    Returns:
        Combined feature map from all circuits
    """
    all_features = []
    
    for circuit in circuits:
        # Measure the circuit
        measurements = measure_quantum_state(circuit, shots)
        
        # Extract features
        features = extract_features_from_measurements(measurements, feature_dim)
        all_features.append(features)
    
    # Stack features (e.g., like channels in a CNN)
    combined_features = np.stack(all_features, axis=-1 if feature_dim else 0)
    return combined_features


def quantum_feature_to_classical_format(
    quantum_features: np.ndarray,
    target_shape: Tuple[int, int, int] = None
) -> np.ndarray:
    """
    Convert quantum features to format suitable for classical neural networks.
    
    Args:
        quantum_features: Array of quantum features
        target_shape: Target shape (height, width, channels) for YOLO input
        
    Returns:
        Reshaped features suitable for YOLO
    """
    # If no target shape is provided, keep as is
    if target_shape is None:
        return quantum_features
    
    # Reshape and potentially rescale features to match expected input
    current_shape = quantum_features.shape
    
    # Handle different dimensional cases
    if len(current_shape) == 1:
        # 1D feature vector - reshape to 2D + channels
        features_reshaped = np.zeros(target_shape)
        flat_length = target_shape[0] * target_shape[1] * target_shape[2]
        
        # Make sure we don't exceed the length of our features
        usable_length = min(len(quantum_features), flat_length)
        features_flat = np.resize(quantum_features, usable_length)
        
        # Fill in the reshaped array
        features_reshaped.flat[:usable_length] = features_flat
        return features_reshaped
        
    elif len(current_shape) == 2:
        # 2D feature map - add channel dimension and resize
        features_with_channel = quantum_features[:, :, np.newaxis]
        
        # Simple resizing by repeating the channel if needed
        if target_shape[2] > 1:
            features_with_channel = np.repeat(features_with_channel, target_shape[2], axis=2)
            
        # Resize spatial dimensions
        from skimage.transform import resize
        resized_features = resize(features_with_channel, target_shape)
        return resized_features
        
    elif len(current_shape) == 3:
        # 3D feature map - resize to target shape
        from skimage.transform import resize
        resized_features = resize(quantum_features, target_shape)
        return resized_features
    
    else:
        raise ValueError(f"Unsupported feature shape: {current_shape}")