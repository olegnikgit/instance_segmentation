#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script demonstrates how to run RF-DETR object detection on a set of images using a custom trained model.

An example to run the script:
python rf_detr_object_detection.py \
    --input_dir /path/to/input/images \
    --dataset folder \
    --max_images 2 \
    --pretrain_weights /path/to/custom/model/checkpoint.pth \
    --output_dir /path/to/output/results \
    --confidence 0.5 \
    --save_masks 1 \
    --save_overlay 1 \


@author: Olegs
"""

# ==========================================================
import os
import argparse
import gc
import torch
import weakref
import cv2
import numpy as np
from sam3.visualization_utils import generate_colors,plot_mask,plot_bbox
from rfdetr import RFDETRNano
from matplotlib import pyplot as plt
import json
from instance_segmentation.database.custom_dataset import CustommDataset


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)

    # add optional dataset argument to allow running on CustommDataset
    parser.add_argument(
        "--dataset",
        choices=["folder", "custom"],
        default="folder",
        help="Type of input dataset: 'folder' for a directory of images, 'custom' for CustommDataset"
    )

    # add optional argument to specify maximum number of images to process (useful for testing)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: None, meaning process all images)"
    )

    # path to pretrain_weights of RF-DETR model
    parser.add_argument("--pretrain_weights", type=str, default="EXPERIMENT_PATH/checkpoint_best_total.pth")

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--confidence", type=float, default=0.5)

    parser.add_argument("--save_masks", type=int, default=1)
    parser.add_argument("--save_overlay", type=int, default=1)

    return parser.parse_args()


# ==========================================================
# SETUP CUDA
# ==========================================================
def cleanup_gpu_memory(obj=None, verbose: bool = False):

    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA is not available. No GPU cleanup needed.")
        return

    def get_memory_stats():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[Before] Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")

    # Ensure we drop all strong references
    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully garbage collected yet.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(f"[After]  Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")


# ==========================================================
# SAVE MASKS
# ==========================================================

def save_masks( image_predictions, mask_dir, base_name ):

    for pred in image_predictions:

        pid = pred["prompt_id"]
        obj_id = pred["object_id"]

        # generate mask from bbox coordinates (assuming bbox is in XYXY format)
        # replace with actual mask if available later
        mask = None
        if "bbox_xyxy" in pred:
            x1, y1, x2, y2 = map(int, pred["bbox_xyxy"])
            mask = np.zeros((pred["image_height"], pred["image_width"]), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

        if mask is not None:
            # Convert mask to uint8 (0 or 255)
            mask_uint8 = (mask * 255).astype("uint8")

            # Save the binary mask as PNG
            mask_path = os.path.join(mask_dir, f"prompt_{pid}", f"{base_name}_obj_{obj_id}.png")
            cv2.imwrite(mask_path, mask_uint8)



# ==========================================================
# SAVE OVERLAY
# ==========================================================
def save_overlay_image(image, predictions, colors, overlay_path):
    """
    Draw bounding boxes and masks from RF-DETR dictionary predictions onto the image.

    Parameters:
    - image: NumPy array of the input image.
    - predictions: List of prediction dicts (the output of collect_rf_detr_predictions).
    - colors: List of colors to use for plotting.
    - overlay_path: Path where the output visualization should be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    h, w = image.shape[:2]

    for obj_id, pred in enumerate(predictions):
        # Unique color picker based on object ID or prompt_id (class_id)
        # Using prompt_id ensures the same classes keep the same color across images
        class_id = pred["prompt_id"]
        color = colors[(class_id * 20 + obj_id) % len(colors)]

        # Create a pseudo-mask from the bounding box for visualization purposes
        x1, y1, x2, y2 = map(int, pred["bbox_xyxy"])
        mask = np.zeros((pred["image_height"], pred["image_width"]), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        # 2. Extract bounding box coordinates
        bbox_xyxy = pred["bbox_xyxy"] # [x1, y1, x2, y2]

        # Text label formatting
        label_text = f"{pred['prompt']} ({pred['score']:.2f})"

        # 3. Plot the pseudo-mask onto the axis
        plot_mask(mask, color=color, ax=ax)

        # 4. Plot the bounding box
        # We wrap the coordinates back into a shape or format plot_bbox expects
        plot_bbox(
            h,
            w,
            bbox_xyxy,  # Passing the [x1, y1, x2, y2] list directly
            text=label_text,
            box_format="XYXY",
            color=color,
            relative_coords=False,
            ax=ax
        )

    ax.axis("off")
    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ==========================================================
# COLLECT PREDICTIONS
# ==========================================================
def collect_rf_detr_predictions(detections, image_path, image_shape, class_names=None):
    """
    Convert RF-DETR sv.Detections outputs into serializable prediction dicts.

    Each prediction dict contains:
    - image_path: str
    - image_name: str
    - image_width: int
    - image_height: int
    - prompt_id: int (Mapped to class_id)
    - prompt: str (Mapped to class name string)
    - object_id: int
    - bbox_xyxy: [x1, y1, x2, y2]
    - bbox_xywh: [x, y, w, h]
    - score: float
    - area: float
    - relative_area: float
    - aspect_ratio: float

    Parameters:
    - detections: sv.Detections object from rfdetr model.predict().
    - image_path: Path to the input image.
    - image_shape: Shape of the input image (H, W, C).
    - class_names: Optional list of strings mapping class IDs to human-readable names.

    Returns:
    - predictions: List of prediction dicts for all detected objects.
    """
    predictions = []
    h_img, w_img = image_shape[:2]
    total_area = float(w_img * h_img)

    # Fallback to class index strings if class_names mapping is not provided
    if class_names is None:
        class_names = []

    # sv.Detections can be iterated to get (xyxy, mask, confidence, class_id, tracker_id, data)
    # Alternatively, we can safely unzip or index its attributes.
    for obj_id in range(len(detections)):
        # Extract basic bounding box and metrics
        x1, y1, x2, y2 = detections.xyxy[obj_id].tolist()
        score = float(detections.confidence[obj_id]) if detections.confidence is not None else 0.0
        class_id = int(detections.class_id[obj_id]) if detections.class_id is not None else 0

        # Calculate bounding box structural metrics
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # Map class names dynamically if available
        prompt_text = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

        pred = {
            # image info
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "image_width": int(w_img),
            "image_height": int(h_img),

            # prompt info (Adapted from class metadata)
            "prompt_id": class_id,
            "prompt": prompt_text,

            # object info
            "object_id": int(obj_id),

            # bbox
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_xywh": [float(x1), float(y1), float(bw), float(bh)],

            # metrics
            "score": score,
            "area": float(area),
            "relative_area": float(area / total_area) if total_area > 0 else 0.0,
            "aspect_ratio": float(bw / bh if bh > 0 else 0.0)
        }

        predictions.append(pred)

    return predictions


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # a subdirectory for binary masks:
    mask_dir = os.path.join( args.output_dir, "masks" )
    # a subdirectory for overlay images:
    overlay_dir = os.path.join( args.output_dir, "overlay" )

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    prompts = ["my_object_name"]  # hard-coded prompt for now, can be extended to accept user-defined prompts in the future

    for pid in range(len(prompts)):
        os.makedirs( os.path.join( mask_dir, f"prompt_{pid}" ), exist_ok=True )


    # get dataset based on argument
    if args.dataset == "folder":  # default: read images from a folder
        valid_ext = ( ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff" )

        image_paths = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(valid_ext)
        ])
    elif args.dataset == "custom":
        dataset = CustommDataset( root_dir=args.input_dir, split="test" )  # use test split by default
        image_paths = [s["image_path"] for s in dataset.selected_samples]


    # optional limit on number of images to process (useful for testing)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images")

    cleanup_gpu_memory(verbose=True)


    model = RFDETRNano(resolution=384,
                       pretrain_weights=args.pretrain_weights)
    # model = RFDETRNano(resolution=320,
    #                    num_queries=50,
    #                    pretrain_weights=args.pretrain_weights)

    model.optimize_for_inference()

    colors = generate_colors(256, 5000)


    # ------------------------------------------------------
    # PROCESS IMAGES
    # ------------------------------------------------------

    # a list to store all predictions if needed for further analysis
    all_predictions = []

    for idx, img_path in enumerate(image_paths):

        base_name = os.path.splitext( os.path.basename(img_path) )[0]

        print( f"\n[{idx+1}/{len(image_paths)}] " f"{os.path.basename(img_path)}" )

        image = cv2.cvtColor( cv2.imread(img_path), cv2.COLOR_BGR2RGB )

        # if args.scale != 1.0:
        #     image = cv2.resize( image, None, fx=args.scale, fy=args.scale )


        # run RF-DETR on the image and get results for all prompts
        detections = model.predict(image, threshold=args.confidence)

        # collect predictions into a serializable format for potential saving or analysis
        image_predictions = collect_rf_detr_predictions(
            detections=detections,
            image_path=img_path,
            image_shape=image.shape,
            class_names=prompts)

        all_predictions.extend(image_predictions)

        # ----------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------

        # print the number of detected objects per prompt
        for pid, prompt in enumerate(prompts):
            num_objects = sum(1 for pred in image_predictions if pred["prompt_id"] == pid)
            print(f"Prompt {pid} ('{prompt}'): {num_objects} objects detected")


        # ----------------------------------------------
        # SAVE MASKS
        # ----------------------------------------------

        if args.save_masks:
            save_masks( image_predictions=image_predictions,
                        mask_dir=mask_dir,
                        base_name=base_name )


        # ----------------------------------------------
        # SAVE OVERLAY
        # ----------------------------------------------

        if args.save_overlay:

            overlay_path = os.path.join( overlay_dir, f"{base_name}.png" )
            save_overlay_image( image=image,
                                predictions=image_predictions,
                                colors=colors,
                                overlay_path=overlay_path )


    # ------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------

    predictions_path = os.path.join( args.output_dir, "predictions.json" )

    with open(predictions_path, "w") as f:

        json.dump( all_predictions, f, indent=2 )

    print( f"\nSaved {len(all_predictions)} predictions to:" )
    print(predictions_path)


    print("\nFinished.")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()


