"""
yolo_utils.py - Utility functions for YOLO object detection model.

This module provides utility functions for bounding box processing, non-maximum suppression,
and evaluation metrics for object detection using the YOLO model.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Any
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def xywh2xyxy(boxes: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: Bounding boxes in [x_center, y_center, width, height] format
        
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format
    """
    if isinstance(boxes, torch.Tensor):
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        x1 = boxes[..., 0] - boxes[..., 2] / 2
        y1 = boxes[..., 1] - boxes[..., 3] / 2
        x2 = boxes[..., 0] + boxes[..., 2] / 2
        y2 = boxes[..., 1] + boxes[..., 3] / 2
        return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy2xywh(boxes: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height] format.
    
    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        
    Returns:
        Bounding boxes in [x_center, y_center, width, height] format
    """
    if isinstance(boxes, torch.Tensor):
        x_center = (boxes[..., 0] + boxes[..., 2]) / 2
        y_center = (boxes[..., 1] + boxes[..., 3]) / 2
        width = boxes[..., 2] - boxes[..., 0]
        height = boxes[..., 3] - boxes[..., 1]
        return torch.stack([x_center, y_center, width, height], dim=-1)
    else:
        x_center = (boxes[..., 0] + boxes[..., 2]) / 2
        y_center = (boxes[..., 1] + boxes[..., 3]) / 2
        width = boxes[..., 2] - boxes[..., 0]
        height = boxes[..., 3] - boxes[..., 1]
        return np.stack([x_center, y_center, width, height], axis=-1)


def box_iou(boxes1: Union[np.ndarray, torch.Tensor], 
           boxes2: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate Intersection over Union (IoU) between two sets of bounding boxes.
    
    Args:
        boxes1: First set of bounding boxes in [x1, y1, x2, y2] format
        boxes2: Second set of bounding boxes in [x1, y1, x2, y2] format
        
    Returns:
        IoU values between each pair of boxes
    """
    if isinstance(boxes1, torch.Tensor):
        # PyTorch implementation
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Expand dimensions for broadcasting
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou
    else:
        # NumPy implementation
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Expand dimensions for broadcasting
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = np.maximum(rb - lt, 0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou


def non_max_suppression(
    boxes: Union[np.ndarray, torch.Tensor],
    scores: Union[np.ndarray, torch.Tensor],
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Perform non-maximum suppression on bounding boxes.
    
    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    # Convert to numpy if tensors
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # Get indices of sorted scores (highest first)
    indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    while indices.size > 0:
        # Keep the box with highest score
        current_index = indices[0]
        keep_indices.append(current_index)
        
        # If only one box left, we're done
        if indices.size == 1:
            break
        
        # Calculate IoU of the kept box with rest
        current_box = boxes[current_index].reshape(1, 4)
        rest_boxes = boxes[indices[1:]]
        
        ious = box_iou(current_box, rest_boxes)[0]
        
        # Keep only boxes with IoU below threshold
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return keep_indices


def process_yolo_output(
    predictions: torch.Tensor,
    anchors: List[List[int]],
    num_classes: int,
    image_size: Tuple[int, int] = (416, 416),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Process raw YOLO output into usable detections.
    
    Args:
        predictions: Raw YOLO prediction tensor
        anchors: Anchor box dimensions
        num_classes: Number of object classes
        image_size: Size of input image (height, width)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of dictionaries with detection results
    """
    # This is a simplified implementation of YOLO output processing
    # In a real scenario, this would depend on specific YOLO version and output format
    
    # Placeholder for actual YOLO output processing
    # Normally, this would decode the YOLO grid outputs to bounding boxes
    # using anchors and apply appropriate transformations
    
    # Simplified placeholder implementation
    batch_size = predictions.shape[0]
    results = []
    
    for i in range(batch_size):
        # Process each image in batch
        # Example placeholder detections
        detections = [
            {
                "bbox": [100, 100, 200, 200],  # [x1, y1, x2, y2]
                "class_id": 0,
                "confidence": 0.9,
                "class_name": "person"
            }
        ]
        results.append(detections)
    
    return results


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: List[str] = None,
    color_map: Dict[int, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw bounding boxes and labels for detections on an image.
    
    Args:
        image: Input image as numpy array (BGR format for OpenCV)
        detections: List of detection dictionaries with bbox, class_id, confidence
        class_names: List of class names for labeling
        color_map: Dictionary mapping class IDs to colors
        
    Returns:
        Image with drawn detections
    """
    # Make a copy to avoid modifying the original
    img_out = image.copy()
    height, width = img_out.shape[:2]
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(100)]  # Arbitrary default names
    
    # Default color map if not provided
    if color_map is None:
        color_map = {}
        for i in range(100):  # Generate colors for up to 100 classes
            color_map[i] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
    
    # Draw each detection
    for det in detections:
        bbox = det["bbox"]
        class_id = det["class_id"]
        conf = det["confidence"]
        
        # Handle different bbox formats, assuming [x1, y1, x2, y2]
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(c) for c in bbox]
        else:
            # Skip invalid bboxes
            continue
        
        # Get color for this class
        color = color_map.get(class_id, (0, 255, 0))  # Default to green
        
        # Draw bounding box
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name}: {conf:.2f}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_out, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img_out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_out


def calculate_map(
    predictions: List[List[Dict]],
    ground_truth: List[List[Dict]],
    iou_threshold: float = 0.5,
    num_classes: int = None
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of predictions for each image, each with bbox, class_id, confidence
        ground_truth: List of ground truth annotations for each image
        iou_threshold: IoU threshold for a detection to be considered correct
        num_classes: Number of object classes
        
    Returns:
        Dictionary with mAP and AP for each class
    """
    # Determine number of classes if not provided
    if num_classes is None:
        class_ids = set()
        for img_preds in predictions:
            for pred in img_preds:
                class_ids.add(pred["class_id"])
        for img_gts in ground_truth:
            for gt in img_gts:
                class_ids.add(gt["class_id"])
        num_classes = max(class_ids) + 1 if class_ids else 0
    
    # Initialize data structures
    # For each class, we'll store all detections across all images
    class_predictions = [[] for _ in range(num_classes)]
    class_ground_truths = [[] for _ in range(num_classes)]
    
    # Collect predictions and ground truths by class
    for img_idx, (img_preds, img_gts) in enumerate(zip(predictions, ground_truth)):
        # Mark all ground truths as not detected yet
        for gt in img_gts:
            gt["detected"] = False
            class_ground_truths[gt["class_id"]].append(gt)
        
        # Add image index to predictions for later reference
        for pred in img_preds:
            pred["image_idx"] = img_idx
            class_predictions[pred["class_id"]].append(pred)
    
    # Calculate AP for each class
    average_precisions = {}
    
    for class_id in range(num_classes):
        # Sort predictions by confidence (descending)
        preds = sorted(class_predictions[class_id], key=lambda x: x["confidence"], reverse=True)
        gts = class_ground_truths[class_id]
        
        # Skip if no ground truths for this class
        if len(gts) == 0:
            average_precisions[class_id] = 0
            continue
        
        # Prepare arrays for precision-recall curve
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # Assign predictions to ground truths
        for i, pred in enumerate(preds):
            img_idx = pred["image_idx"]
            
            # Get ground truths for this image
            img_gts = [gt for gt in gts if gt.get("image_idx", -1) == img_idx]
            
            best_iou = -np.inf
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j, gt in enumerate(img_gts):
                if gt["detected"]:
                    continue
                
                # Calculate IoU
                pred_box = np.array(pred["bbox"]).reshape(1, 4)
                gt_box = np.array(gt["bbox"]).reshape(1, 4)
                iou = box_iou(pred_box, gt_box)[0][0]
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if detection is correct using IoU threshold
            if best_iou >= iou_threshold:
                # Mark ground truth as detected
                img_gts[best_gt_idx]["detected"] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute cumulative sums for precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / len(gts)
        
        # Add sentinel values
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # Find points where recall increases
        indices = np.where(recall[1:] != recall[:-1])[0] + 1
        
        # Calculate average precision
        ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
        average_precisions[class_id] = ap
    
    # Calculate mAP
    mean_ap = np.mean([ap for ap in average_precisions.values()])
    
    return {
        "mAP": mean_ap,
        "AP": average_precisions
    }


def save_yolo_annotation(
    bboxes: List[List[float]],
    class_ids: List[int],
    image_width: int,
    image_height: int,
    output_path: str
) -> None:
    """
    Save object detection annotations in YOLO format.
    
    Args:
        bboxes: List of bounding boxes in [x1, y1, x2, y2] format
        class_ids: List of class IDs for each bounding box
        image_width: Width of the image
        image_height: Height of the image
        output_path: Path to save the annotation file
    """
    with open(output_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_ids):
            # Convert [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
            # All values normalized to [0, 1]
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            # Write to file: class_id x_center y_center width height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def load_yolo_annotation(
    annotation_path: str,
    image_width: int,
    image_height: int
) -> Tuple[List[List[float]], List[int]]:
    """
    Load object detection annotations from YOLO format.
    
    Args:
        annotation_path: Path to the annotation file
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Tuple of (bboxes, class_ids) where bboxes are in [x1, y1, x2, y2] format
    """
    bboxes = []
    class_ids = []
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                
                # Convert to [x1, y1, x2, y2] format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                bboxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
    except Exception as e:
        print(f"Error loading annotation file {annotation_path}: {e}")
    
    return bboxes, class_ids


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """
    Visualize object detections using matplotlib.
    
    Args:
        image: Input image (RGB format)
        detections: List of detection dictionaries
        class_names: List of class names
        figsize: Figure size for matplotlib
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display image
    ax.imshow(image)
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(100)]
    
    # Add each bounding box
    for det in detections:
        bbox = det["bbox"]
        class_id = det["class_id"]
        confidence = det["confidence"]
        
        # Get class name
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Get random color for this class (consistent across detections)
        np.random.seed(class_id)
        color = np.random.rand(3)
        
        # Create rectangle patch
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name}: {confidence:.2f}"
        ax.text(x1, y1 - 5, label, color=color, fontsize=10, backgroundcolor='white')
    
    # Hide axes and show plot
    ax.axis('off')
    plt.tight_layout()
    plt.show()