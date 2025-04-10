"""
BRQI (Bit-Plane Representation of Quantum Images) encoding module.
This module handles the conversion from classical images to BRQI representation.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def binary_to_gray_code(n):
    """
    Convert a binary number to Gray code.
    
    Args:
        n (int): Binary number
        
    Returns:
        int: Gray code
    """
    return n ^ (n >> 1)

def calculate_qubits_needed(image_size):
    """
    Calculate the number of qubits needed to represent the position (coordinates).
    
    Args:
        image_size (tuple): Size of the image (width, height)
        
    Returns:
        tuple: (qubits_for_width, qubits_for_height)
    """
    width, height = image_size
    
    # Calculate the number of qubits needed for position encoding
    qubits_width = math.ceil(math.log2(width))
    qubits_height = math.ceil(math.log2(height))
    
    return qubits_width, qubits_height

def create_brqi_circuit(bitplane, image_size, use_gray_code=True):
    """
    Create a quantum circuit for BRQI representation of a single bitplane.
    
    Args:
        bitplane (numpy.ndarray): 2D array representing a bit plane (0s and 1s)
        image_size (tuple): Size of the image (width, height)
        use_gray_code (bool): Whether to use Gray code for position encoding
        
    Returns:
        qiskit.QuantumCircuit: Quantum circuit with BRQI encoding
    """
    width, height = image_size
    
    # Calculate qubits needed for position encoding
    qubits_width, qubits_height = calculate_qubits_needed(image_size)
    total_position_qubits = qubits_width + qubits_height
    
    # Create quantum registers
    position_qr = QuantumRegister(total_position_qubits, 'position')
    color_qr = QuantumRegister(1, 'color')
    cr = ClassicalRegister(total_position_qubits + 1, 'measurement')
    
    # Create quantum circuit
    circuit = QuantumCircuit(position_qr, color_qr, cr)
    
    # Put all position qubits in superposition
    for i in range(total_position_qubits):
        circuit.h(position_qr[i])
    
    # Encode the bit plane values
    for y in range(min(height, 2**qubits_height)):
        for x in range(min(width, 2**qubits_width)):
            # Skip if out of bounds
            if y >= bitplane.shape[0] or x >= bitplane.shape[1]:
                continue
            
            # Convert position to binary representation
            x_bin = format(x, f'0{qubits_width}b')
            y_bin = format(y, f'0{qubits_height}b')
            
            # If using Gray code, convert
            if use_gray_code:
                x_gray = format(binary_to_gray_code(x), f'0{qubits_width}b')
                y_gray = format(binary_to_gray_code(y), f'0{qubits_height}b')
                position_str = x_gray + y_gray
            else:
                position_str = x_bin + y_bin
            
            # Create multi-controlled X gate to encode the bit value
            # If the bit value is 1, apply X to the color qubit
            if bitplane[y, x] == 1:
                controls = []
                for i, bit in enumerate(position_str):
                    if bit == '0':
                        circuit.x(position_qr[i])  # Flip for control on |0⟩
                        controls.append(i)
                
                # Apply multi-controlled X gate
                circuit.mcx(
                    [position_qr[i] for i in range(total_position_qubits)],
                    color_qr[0]
                )
                
                # Flip back the qubits
                for i in controls:
                    circuit.x(position_qr[i])
    
    return circuit

def encode_image_brqi(bitplanes, image_size):
    """
    Encode all bitplanes of an image into BRQI representation.
    
    Args:
        bitplanes (dict): Dictionary containing bit planes for each channel
        image_size (tuple): Size of the image (width, height)
        
    Returns:
        dict: Dictionary of quantum circuits for each bitplane
    """
    logger.info(f"Encoding image of size {image_size} into BRQI representation")
    
    circuits = {}
    
    for channel_name, channel_bitplanes in bitplanes.items():
        logger.info(f"Processing channel: {channel_name}")
        channel_circuits = {}
        
        for bit_name, bitplane in channel_bitplanes.items():
            logger.info(f"Creating circuit for {channel_name} - {bit_name}")
            circuit = create_brqi_circuit(bitplane, image_size)
            channel_circuits[bit_name] = circuit
        
        circuits[channel_name] = channel_circuits
    
    return circuits

def simulate_brqi_circuit(circuit):
    """
    Simulate a BRQI circuit to get the quantum state.
    
    Args:
        circuit (qiskit.QuantumCircuit): Quantum circuit with BRQI encoding
        
    Returns:
        qiskit.quantum_info.Statevector: Quantum state
    """
    # Get the statevector from the circuit
    statevector = Statevector.from_instruction(circuit)
    return statevector

def convert_small_image_to_brqi(image, max_size=(32, 32)):
    """
    Convert a small image to BRQI representation.
    For larger images, use downsizing or tiling approaches.
    
    Args:
        image (numpy.ndarray): Input image
        max_size (tuple): Maximum size for direct BRQI encoding
        
    Returns:
        dict: Dictionary of BRQI circuits
    """
    # Ensure image is smaller than max_size
    height, width = image.shape[:2]
    if height > max_size[1] or width > max_size[0]:
        logger.warning(f"Image size ({width}x{height}) exceeds maximum size {max_size}. Resizing...")
        from ..preprocessing.image_preprocessing import preprocess_image
        image = preprocess_image(image, target_size=max_size)
    
    # Extract bitplanes
    from ..preprocessing.image_preprocessing import extract_bitplanes
    bitplanes = extract_bitplanes(image)
    
    # Create BRQI circuits
    circuits = encode_image_brqi(bitplanes, image.shape[:2])
    
    return circuits

def simplify_brqi_for_limited_qubits(image, max_qubits=20):
    """
    Simplify BRQI encoding to handle limited qubits.
    This function implements strategies to reduce qubit requirements.
    
    Args:
        image (numpy.ndarray): Input image
        max_qubits (int): Maximum number of qubits available
        
    Returns:
        dict: Dictionary of BRQI circuits
    """
    # Calculate current requirements
    height, width = image.shape[:2]
    qubits_width, qubits_height = calculate_qubits_needed((width, height))
    total_position_qubits = qubits_width + qubits_height
    
    # Add 1 for color qubit
    total_qubits_needed = total_position_qubits + 1
    
    if total_qubits_needed <= max_qubits:
        # We can use standard BRQI
        logger.info(f"Standard BRQI requires {total_qubits_needed} qubits, which is under the limit of {max_qubits}")
        return convert_small_image_to_brqi(image)
    
    # We need to simplify
    logger.info(f"Standard BRQI would require {total_qubits_needed} qubits, exceeding limit of {max_qubits}")
    
    # Strategy 1: Downsample the image
    max_position_qubits = max_qubits - 1  # Reserve 1 qubit for color
    max_dim = 2 ** (max_position_qubits // 2)
    
    new_size = (min(width, max_dim), min(height, max_dim))
    logger.info(f"Downsampling image to {new_size}")
    
    from ..preprocessing.image_preprocessing import preprocess_image
    downsampled_image = preprocess_image(image, target_size=new_size)
    
    return convert_small_image_to_brqi(downsampled_image)

def encode_batch_brqi(images, max_qubits=None):
    """
    Encode a batch of images into BRQI representation.
    
    Args:
        images (list): List of input images
        max_qubits (int, optional): Maximum number of qubits available
        
    Returns:
        list: List of dictionaries of BRQI circuits
    """
    brqi_batch = []
    
    for img in images:
        if max_qubits is not None:
            brqi = simplify_brqi_for_limited_qubits(img, max_qubits)
        else:
            # Use the default conversion, which may resize very large images
            brqi = convert_small_image_to_brqi(img)
        
        brqi_batch.append(brqi)
    
    return brqi_batch

if __name__ == "__main__":
    # Test BRQI encoding
    print("Testing BRQI encoding...")
    
    # Create a small test image (4x4)
    test_img = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.uint8)
    
    # Create a bitplane representation
    bitplane = test_img
    
    # Create BRQI circuit
    circuit = create_brqi_circuit(bitplane, test_img.shape)
    
    print("BRQI circuit created successfully:")
    print(circuit)
    
    # Simulate circuit
    state = simulate_brqi_circuit(circuit)
    print(f"Circuit generates a state with {len(state)} amplitudes")
    
    print("BRQI encoding test completed.")