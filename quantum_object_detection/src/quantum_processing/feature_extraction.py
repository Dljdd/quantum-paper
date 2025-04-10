"""
Quantum Feature Extraction module for object detection.
This module defines quantum circuits for extracting features from BRQI-encoded images.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit.circuit.library as library
from qiskit import transpile, Aer
from qiskit.visualization import plot_histogram
import pennylane as qml
import math
import logging
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumFeatureExtractor:
    """Class for quantum feature extraction from BRQI-encoded images."""
    
    def __init__(self, n_position_qubits: int, feature_map_type: str = "ZZFeatureMap"):
        """
        Initialize quantum feature extractor.
        
        Args:
            n_position_qubits (int): Number of position qubits in BRQI encoding
            feature_map_type (str): Type of feature map to use
        """
        self.n_position_qubits = n_position_qubits
        self.n_color_qubits = 1  # BRQI typically uses one qubit for color
        self.total_qubits = n_position_qubits + self.n_color_qubits
        self.feature_map_type = feature_map_type
        
        logger.info(f"Initialized QuantumFeatureExtractor with {self.total_qubits} qubits")
        logger.info(f"Using feature map: {feature_map_type}")
    
    def _create_feature_map(self) -> QuantumCircuit:
        """
        Create a feature map circuit for feature extraction.
        
        Returns:
            QuantumCircuit: Qiskit quantum circuit for feature mapping
        """
        if self.feature_map_type == "ZZFeatureMap":
            # Create a ZZFeatureMap - good for detecting correlations
            feature_map = library.ZZFeatureMap(
                feature_dimension=self.total_qubits,
                reps=2,
                entanglement='linear'
            )
        elif self.feature_map_type == "PauliFeatureMap":
            # Create a PauliFeatureMap - more expressive feature map
            feature_map = library.PauliFeatureMap(
                feature_dimension=self.total_qubits,
                reps=2,
                paulis=['Z', 'Y']
            )
        elif self.feature_map_type == "QConv":
            # Create a custom quantum convolutional feature map
            feature_map = self._create_quantum_conv()
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
        
        return feature_map
    
    def _create_quantum_conv(self) -> QuantumCircuit:
        """
        Create a quantum convolutional feature map.
        
        Returns:
            QuantumCircuit: Quantum convolutional circuit
        """
        qc = QuantumCircuit(self.total_qubits)
        
        # Apply Hadamard gates to create superposition
        for i in range(self.total_qubits):
            qc.h(i)
        
        # Apply "convolutional" gates in a sliding window fashion
        for i in range(self.total_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(np.pi / 4, i + 1)
            qc.cx(i, i + 1)
        
        # Add another layer of "convolution"
        for i in range(self.total_qubits - 2):
            qc.cx(i, i + 2)
            qc.rz(np.pi / 4, i + 2)
            qc.cx(i, i + 2)
        
        return qc
    
    def _create_variational_circuit(self, n_layers: int = 2) -> QuantumCircuit:
        """
        Create a variational circuit for trainable feature extraction.
        
        Args:
            n_layers (int): Number of variational layers
            
        Returns:
            QuantumCircuit: Parameterized variational circuit
        """
        qc = QuantumCircuit(self.total_qubits)
        
        # Number of parameters per layer
        n_params_per_layer = self.total_qubits * 2  # Rx and Rz for each qubit
        
        # Create trainable parameters
        params = np.random.random(n_layers * n_params_per_layer) * 2 * np.pi
        
        param_idx = 0
        for layer in range(n_layers):
            # Rotation gates with trainable parameters
            for i in range(self.total_qubits):
                qc.rx(params[param_idx], i)
                param_idx += 1
                qc.rz(params[param_idx], i)
                param_idx += 1
            
            # Entangling gates
            for i in range(self.total_qubits - 1):
                qc.cx(i, i + 1)
            
            # Connect the last qubit to the first to form a cycle
            if self.total_qubits > 2:
                qc.cx(self.total_qubits - 1, 0)
        
        return qc, params
        
    def create_feature_extraction_circuit(self, add_measurement: bool = True) -> QuantumCircuit:
        """
        Create a complete feature extraction circuit.
        
        Args:
            add_measurement (bool): Whether to add measurement operations
            
        Returns:
            QuantumCircuit: Feature extraction circuit
        """
        # Create quantum registers
        qr = QuantumRegister(self.total_qubits, 'q')
        if add_measurement:
            cr = ClassicalRegister(self.total_qubits, 'c')
            qc = QuantumCircuit(qr, cr)
        else:
            qc = QuantumCircuit(qr)
        
        # Add feature map
        feature_map = self._create_feature_map()
        qc = qc.compose(feature_map)
        
        # Add variational circuit for trainable features
        var_circuit, _ = self._create_variational_circuit(n_layers=2)
        qc = qc.compose(var_circuit)
        
        # Add measurements if requested
        if add_measurement:
            qc.measure(qr, cr)
        
        return qc
    
    def extract_features(self, brqi_circuit: QuantumCircuit, n_shots: int = 1024) -> np.ndarray:
        """
        Extract features from a BRQI-encoded circuit.
        
        Args:
            brqi_circuit (QuantumCircuit): BRQI-encoded circuit
            n_shots (int): Number of measurement shots
            
        Returns:
            numpy.ndarray: Extracted feature vector
        """
        # Create feature extraction circuit
        extraction_circuit = self.create_feature_extraction_circuit(add_measurement=False)
        
        # Combine BRQI encoding with feature extraction
        # First, remove measurement from BRQI circuit if any
        if hasattr(brqi_circuit, 'remove_final_measurements'):
            brqi_without_measure = brqi_circuit.remove_final_measurements()
        else:
            brqi_without_measure = brqi_circuit
        
        # Combine circuits
        combined_circuit = brqi_without_measure.compose(extraction_circuit)
        
        # Add measurement to all qubits
        cr = ClassicalRegister(self.total_qubits, 'c')
        combined_circuit.add_register(cr)
        combined_circuit.measure_all()
        
        # Run simulation
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(combined_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=n_shots).result()
        counts = result.get_counts()
        
        # Convert counts to feature vector
        feature_vector = self._counts_to_feature_vector(counts, n_shots)
        
        return feature_vector
    
    def _counts_to_feature_vector(self, counts: dict, n_shots: int) -> np.ndarray:
        """
        Convert measurement counts to feature vector.
        
        Args:
            counts (dict): Measurement counts
            n_shots (int): Number of shots used
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Sort by bit string to ensure consistent ordering
        sorted_counts = sorted(counts.items())
        
        # Convert to probabilities
        probabilities = [count / n_shots for bit_string, count in sorted_counts]
        
        # Create feature vector
        feature_vector = np.array(probabilities)
        
        return feature_vector
    
    def extract_features_batch(self, brqi_circuits_list: List[Dict], n_shots: int = 1024) -> List[Dict]:
        """
        Extract features from a batch of BRQI-encoded circuits.
        
        Args:
            brqi_circuits_list (List[Dict]): List of dictionaries containing BRQI circuits
            n_shots (int): Number of measurement shots
            
        Returns:
            List[Dict]: List of dictionaries containing extracted features
        """
        features_batch = []
        
        for brqi_circuits in brqi_circuits_list:
            features_dict = {}
            
            for channel_name, channel_circuits in brqi_circuits.items():
                channel_features = {}
                
                for bit_name, circuit in channel_circuits.items():
                    logger.info(f"Extracting features for {channel_name} - {bit_name}")
                    features = self.extract_features(circuit, n_shots)
                    channel_features[bit_name] = features
                
                features_dict[channel_name] = channel_features
            
            features_batch.append(features_dict)
        
        return features_batch

class QuantumEdgeDetector:
    """
    Quantum circuit for edge detection in BRQI-encoded images.
    """
    
    def __init__(self, n_position_qubits: int):
        """
        Initialize quantum edge detector.
        
        Args:
            n_position_qubits (int): Number of position qubits in BRQI encoding
        """
        self.n_position_qubits = n_position_qubits
        self.n_color_qubits = 1
        self.total_qubits = n_position_qubits + self.n_color_qubits
        
    def create_edge_detection_circuit(self) -> QuantumCircuit:
        """
        Create a quantum circuit for edge detection.
        
        Returns:
            QuantumCircuit: Edge detection circuit
        """
        qc = QuantumCircuit(self.total_qubits)
        
        # Apply Hadamard gates to position qubits to create superposition
        for i in range(self.n_position_qubits):
            qc.h(i)
        
        # Apply edge detection operations
        # This is a simplified version - a real implementation would
        # use more sophisticated quantum operations
        
        # Apply CX gates between adjacent position qubits
        for i in range(self.n_position_qubits - 1):
            qc.cx(i, i + 1)
        
        # Apply controlled-Z gate from the last position qubit to the color qubit
        qc.cz(self.n_position_qubits - 1, self.total_qubits - 1)
        
        return qc
    
    def detect_edges(self, brqi_circuit: QuantumCircuit, n_shots: int = 1024) -> np.ndarray:
        """
        Detect edges in a BRQI-encoded image.
        
        Args:
            brqi_circuit (QuantumCircuit): BRQI-encoded circuit
            n_shots (int): Number of measurement shots
            
        Returns:
            numpy.ndarray: Edge map
        """
        # Create edge detection circuit
        edge_circuit = self.create_edge_detection_circuit()
        
        # Combine BRQI encoding with edge detection
        if hasattr(brqi_circuit, 'remove_final_measurements'):
            brqi_without_measure = brqi_circuit.remove_final_measurements()
        else:
            brqi_without_measure = brqi_circuit
        
        combined_circuit = brqi_without_measure.compose(edge_circuit)
        
        # Add measurement to all qubits
        cr = ClassicalRegister(self.total_qubits, 'c')
        combined_circuit.add_register(cr)
        combined_circuit.measure_all()
        
        # Run simulation
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(combined_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=n_shots).result()
        counts = result.get_counts()
        
        # Convert counts to edge map
        edge_map = self._counts_to_edge_map(counts)
        
        return edge_map
    
    def _counts_to_edge_map(self, counts: dict) -> np.ndarray:
        """
        Convert measurement counts to edge map.
        
        Args:
            counts (dict): Measurement counts
            
        Returns:
            numpy.ndarray: Edge map
        """
        # Extract position qubits for edge map
        width = height = 2 ** (self.n_position_qubits // 2)
        edge_map = np.zeros((height, width))
        
        total_counts = sum(counts.values())
        
        for bit_string, count in counts.items():
            # Extract position bits
            position_bits = bit_string[:self.n_position_qubits]
            
            # Convert to x, y coordinates
            x_bits = position_bits[:self.n_position_qubits // 2]
            y_bits = position_bits[self.n_position_qubits // 2:self.n_position_qubits]
            
            x = int(x_bits, 2)
            y = int(y_bits, 2)
            
            # Skip if outside bounds
            if x >= width or y >= height:
                continue
            
            # Update edge map with probability
            edge_map[y, x] = count / total_counts
        
        # Normalize to [0, 1]
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        
        return edge_map

class ParameterizedQuantumCircuit:
    """
    Parameterized quantum circuit for trainable feature extraction.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        """
        Initialize parameterized quantum circuit.
        
        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * n_layers * 2  # 2 parameters (Rx, Rz) per qubit per layer
        
        # Initialize random parameters
        self.params = np.random.random(self.n_params) * 2 * np.pi
        
    def create_circuit(self) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        Create a parameterized quantum circuit.
        
        Returns:
            Tuple[QuantumCircuit, np.ndarray]: Circuit and parameters
        """
        qc = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation gates with trainable parameters
            for i in range(self.n_qubits):
                qc.rx(self.params[param_idx], i)
                param_idx += 1
                qc.rz(self.params[param_idx], i)
                param_idx += 1
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Connect the last qubit to the first to form a cycle
            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)
        
        return qc, self.params
    
    def set_parameters(self, new_params: np.ndarray):
        """
        Set new parameters for the circuit.
        
        Args:
            new_params (numpy.ndarray): New parameters
        """
        if len(new_params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(new_params)}")
        
        self.params = new_params
    
    def get_parameters(self) -> np.ndarray:
        """
        Get current parameters.
        
        Returns:
            numpy.ndarray: Current parameters
        """
        return self.params

class QuantumFeatureTrainer:
    """
    Trainer for quantum feature extraction circuits.
    """
    
    def __init__(self, pqc: ParameterizedQuantumCircuit, learning_rate: float = 0.01):
        """
        Initialize quantum feature trainer.
        
        Args:
            pqc (ParameterizedQuantumCircuit): Parameterized quantum circuit
            learning_rate (float): Learning rate for gradient descent
        """
        self.pqc = pqc
        self.learning_rate = learning_rate
    
    def _quantum_gradient(self, params: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        Compute quantum gradient using parameter shift rule.
        
        Args:
            params (numpy.ndarray): Current parameters
            input_data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Gradient
        """
        # This is a simplified version - real implementation would use
        # quantum gradient techniques like parameter shift rule
        
        # Create a device using PennyLane
        dev = qml.device("default.qubit", wires=self.pqc.n_qubits)
        
        # Define a quantum function
        @qml.qnode(dev)
        def circuit(parameters, inputs):
            # Encode inputs
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i)
            
            # Variational part
            param_idx = 0
            for layer in range(self.pqc.n_layers):
                for i in range(self.pqc.n_qubits):
                    qml.RX(parameters[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(parameters[param_idx], wires=i)
                    param_idx += 1
                
                for i in range(self.pqc.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                if self.pqc.n_qubits > 2:
                    qml.CNOT(wires=[self.pqc.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.pqc.n_qubits)]
        
        # Compute gradient
        gradient = qml.grad(circuit)(params, input_data)
        
        return np.array(gradient)
    
    def train_step(self, input_data: np.ndarray, target_data: np.ndarray) -> float:
        """
        Perform a single training step.
        
        Args:
            input_data (numpy.ndarray): Input data
            target_data (numpy.ndarray): Target data (for supervised training)
            
        Returns:
            float: Loss value
        """
        # Get current parameters
        params = self.pqc.get_parameters()
        
        # Compute gradient
        gradient = self._quantum_gradient(params, input_data)
        
        # Update parameters
        new_params = params - self.learning_rate * gradient
        self.pqc.set_parameters(new_params)
        
        # Compute loss (simplified)
        loss = np.mean((target_data - self._predict(input_data)) ** 2)
        
        return loss
    
    def _predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make a prediction with current parameters.
        
        Args:
            input_data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Prediction
        """
        # Get current parameters
        params = self.pqc.get_parameters()
        
        # Create a device using PennyLane
        dev = qml.device("default.qubit", wires=self.pqc.n_qubits)
        
        # Define a quantum function
        @qml.qnode(dev)
        def circuit(parameters, inputs):
            # Encode inputs
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i)
            
            # Variational part
            param_idx = 0
            for layer in range(self.pqc.n_layers):
                for i in range(self.pqc.n_qubits):
                    qml.RX(parameters[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(parameters[param_idx], wires=i)
                    param_idx += 1
                
                for i in range(self.pqc.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                if self.pqc.n_qubits > 2:
                    qml.CNOT(wires=[self.pqc.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.pqc.n_qubits)]
        
        # Compute prediction
        prediction = circuit(params, input_data)
        
        return np.array(prediction)
    
    def train(
        self, 
        input_data: np.ndarray, 
        target_data: np.ndarray, 
        n_epochs: int = 100,
        batch_size: int = 4,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the quantum feature extraction circuit.
        
        Args:
            input_data (numpy.ndarray): Input data
            target_data (numpy.ndarray): Target data
            n_epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (bool): Whether to print progress
            
        Returns:
            List[float]: Loss history
        """
        n_samples = len(input_data)
        loss_history = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            # Create batches
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                if len(batch_indices) < batch_size:
                    continue  # Skip incomplete batches
                
                batch_input = input_data[batch_indices]
                batch_target = target_data[batch_indices]
                
                # Train on batch
                batch_loss = self.train_step(batch_input, batch_target)
                epoch_loss += batch_loss * len(batch_indices)
            
            # Compute average loss
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.6f}")
        
        return loss_history

if __name__ == "__main__":
    # Test feature extraction
    print("Testing quantum feature extraction...")
    
    # Create a simple BRQI circuit for testing
    from brqi_encoding import create_brqi_circuit
    import numpy as np
    
    # Create a small test image (4x4)
    test_img = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.uint8)
    
    # Create BRQI circuit
    brqi_circuit = create_brqi_circuit(test_img, test_img.shape)
    
    # Create feature extractor
    n_position_qubits = 4  # 2 for width, 2 for height
    feature_extractor = QuantumFeatureExtractor(n_position_qubits)
    
    # Extract features
    features = feature_extractor.extract_features(brqi_circuit, n_shots=1024)
    
    print(f"Extracted feature vector shape: {features.shape}")
    
    # Test edge detector
    edge_detector = QuantumEdgeDetector(n_position_qubits)
    edge_map = edge_detector.detect_edges(brqi_circuit)
    
    print(f"Edge map shape: {edge_map.shape}")
    
    print("Quantum feature extraction test completed.")