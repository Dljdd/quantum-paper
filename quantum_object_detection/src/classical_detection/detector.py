"""
detector.py - Integration module between quantum features and YOLO object detection.

This module provides the main interface for object detection using quantum-extracted features
with a classical YOLO model.
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Union, Any, Optional
import os
import time

from src.quantum_processing.measurement import quantum_feature_to_classical_format
from src.classical_detection.yolo_model import load_yolo_model, convert_quantum_features_to_tensor
from src.classical_detection.yolo_utils import (
    process_yolo_output, 
    non_max_suppression, 
    draw_detections,
    calculate_map
)


class QuantumYOLODetector:
    """
    Main detector class integrating quantum features with YOLO model.
    """
    
    def __init__(
        self,
        model_type: str = "yolov5s",
        num_classes: int = 80,
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        pretrained_weights: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the quantum-classical hybrid detector.
        
        Args:
            model_type: Type of YOLO model to use
            num_classes: Number of object classes
            class_names: List of class names (optional)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            pretrained_weights: Path to pretrained weights file (optional)
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # Set default quantum feature shape
        self.quantum_feature_shape = (416, 416, 3)
        
        # Set class names
        self.class_names = class_names if class_names else [f"class_{i}" for i in range(num_classes)]
        
        # Load YOLO model
        print(f"Loading {model_type} on {device}...")
        self.model = load_yolo_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained_weights=pretrained_weights,
            quantum_feature_shape=self.quantum_feature_shape
        ).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        print("Model loaded successfully")
        
        # Initialize color map for visualization
        self.color_map = {}
        np.random.seed(42)  # For consistent colors
        for i in range(num_classes):
            self.color_map[i] = tuple(map(int, np.random.randint(0, 255, size=3)))
    
    def detect(
        self,
        quantum_features: np.ndarray,
        original_image_shape: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection using quantum features.
        
        Args:
            quantum_features: Quantum-extracted features
            original_image_shape: Original image shape for scaling detections (optional)
            
        Returns:
            List of detection dictionaries with bbox, class_id, confidence
        """
        # Convert quantum features to format suitable for YOLO
        yolo_features = quantum_feature_to_classical_format(
            quantum_features,
            target_shape=self.quantum_feature_shape
        )
        
        # Convert to tensor
        features_tensor = convert_quantum_features_to_tensor(yolo_features)
        features_tensor = features_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(features_tensor)
            inference_time = time.time() - start_time
            
        print(f"Inference completed in {inference_time:.3f} seconds")
        
        # Process predictions
        # In a real implementation, this would use the specific decoding logic
        # for the YOLO version being used
        detections = self._process_predictions(predictions, original_image_shape)
        
        return detections
    
    def _process_predictions(
        self,
        predictions: torch.Tensor,
        original_image_shape: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process raw YOLO predictions into usable detections.
        
        Args:
            predictions: Raw YOLO prediction tensor
            original_image_shape: Original image shape for scaling detections
            
        Returns:
            List of detection dictionaries
        """
        # Example YOLO output processing
        # This is a placeholder - the actual implementation would depend
        # on the specific YOLO version and output format
        
        # Example: process first image in batch
        pred = predictions[0].cpu()
        
        # Get dimensions
        height, width = self.quantum_feature_shape[:2]
        
        # Process predictions to get bounding boxes, classes, and scores
        # This is simplified and would need to be adapted to the actual YOLO output format
        # In a real implementation, this would involve:
        # 1. Decoding grid cell predictions to bounding boxes
        # 2. Applying confidence threshold
        # 3. Performing NMS
        
        # Placeholder implementation:
        # The shape and format would depend on your specific YOLO implementation
        boxes = []
        scores = []
        class_ids = []
        
        # Example detection for demonstration (this should be replaced with actual logic)
        example_box = torch.tensor([0.2 * width, 0.3 * height, 0.5 * width, 0.7 * height])  # x1, y1, x2, y2
        example_score = torch.tensor(0.8)
        example_class = torch.tensor(0)  # First class
        
        boxes.append(example_box)
        scores.append(example_score)
        class_ids.append(example_class)
        
        # Convert to numpy
        boxes = torch.stack(boxes).numpy() if boxes else np.array([])
        scores = torch.stack(scores).numpy() if scores else np.array([])
        class_ids = torch.stack(class_ids).numpy() if class_ids else np.array([])
        
        # Scale boxes to original image size if provided
        if original_image_shape is not None:
            orig_height, orig_width = original_image_shape
            scale_x = orig_width / width
            scale_y = orig_height / height
            
            # Apply scaling
            if len(boxes) > 0:
                boxes[:, 0] *= scale_x  # x1
                boxes[:, 2] *= scale_x  # x2
                boxes[:, 1] *= scale_y  # y1
                boxes[:, 3] *= scale_y  # y2
        
        # Format results
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append({
                "bbox": box.tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f"class_{class_id}"
            })
        
        return detections
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on the original image.
        
        Args:
            image: Original image as numpy array
            detections: List of detection dictionaries
            show: Whether to display the image
            save_path: Path to save the output image (optional)
            
        Returns:
            Image with drawn detections
        """
        # Draw detections on image
        output_image = draw_detections(
            image=image,
            detections=detections,
            class_names=self.class_names,
            color_map=self.color_map
        )
        
        # Display if requested
        if show:
            # Convert BGR to RGB for display
            display_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            # Use matplotlib to show image
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(display_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, output_image)
            print(f"Detection image saved to {save_path}")
        
        return output_image
    
    def evaluate(
        self,
        quantum_features_list: List[np.ndarray],
        ground_truth_annotations: List[List[Dict]],
        original_image_shapes: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate detector performance using ground truth annotations.
        
        Args:
            quantum_features_list: List of quantum features for each image
            ground_truth_annotations: List of ground truth annotations for each image
            original_image_shapes: List of original image shapes (optional)
            
        Returns:
            Dictionary with evaluation metrics (mAP, etc.)
        """
        all_predictions = []
        
        # Get detections for each image
        for i, features in enumerate(quantum_features_list):
            # Get original image shape if available
            orig_shape = original_image_shapes[i] if original_image_shapes else None
            
            # Detect objects
            detections = self.detect(features, orig_shape)
            all_predictions.append(detections)
        
        # Calculate mAP
        metrics = calculate_map(
            predictions=all_predictions,
            ground_truth=ground_truth_annotations,
            iou_threshold=0.5,
            num_classes=self.num_classes
        )
        
        return metrics


def get_anchors(
    anchors_path: str = "data/anchors.txt"
) -> List[List[int]]:
    """
    Load anchor box dimensions from file.
    
    Args:
        anchors_path: Path to anchors file
        
    Returns:
        List of anchor box dimensions [width, height]
    """
    try:
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    except Exception as e:
        print(f"Error loading anchors: {e}")
        # Return default anchors
        return np.array([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ])