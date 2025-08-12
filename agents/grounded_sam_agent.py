# grounded_sam_agent.py
# Modified from grounded_sam_simple_demo.py in original Multi-Agent-VQA repo

import cv2
import numpy as np
import supervision as sv
import re
import torch
import torchvision
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('./tools/grounded-sam/Grounded-Segment-Anything')

from segment_anything.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from GroundingDINO.groundingdino.util.inference import annotate, Model

def plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename):
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[:, :, [2, 1, 0]]  # BGR2RGB
    plt.imsave('bboxes' + filename + '.jpg', annotated_frame)

def query_grounding_dino(device, args, model, image_path, text_prompt="bear.", save_image=False):
    """
    Run GroundingDINO object detection given an image and text prompt.
    """
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model.to(device),
        image=image,
        caption=text_prompt,
        box_threshold=args['dino']['BOX_THRESHOLD'],
        text_threshold=args['dino']['TEXT_THRESHOLD'],
        device=device
    )

    if save_image:
        match = re.search(r'n(\d+)\.jpg', image_path)
        filename = match.group(1) if match else "output"
        print(f"Saving filename: {filename}")
        plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename)

    # Convert boxes to absolute coordinates
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])

    return image_source, boxes, logits, phrases

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # ax.imshow(img)
    plt.imsave('masks.jpg', img)


def query_sam(device, args, sam_mask_generator, image):
    """
    Generate segmentation masks for the entire image.
    """
    masks = sam_mask_generator.generate(image)
    show_anns(masks)
    return masks

class GroundedSAMAgent:
    def __init__(self,
                 grounding_dino_config,
                 grounding_dino_ckpt,
                 sam_encoder_version=None,
                 sam_ckpt=None,
                 device=None):

        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Init GroundingDINO
        self.grounding_dino_model = Model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_ckpt
        )

        # Init SAM only if segmentation params are given
        if sam_encoder_version and sam_ckpt:
            sam = sam_model_registry[sam_encoder_version](checkpoint=sam_ckpt)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
        else:
            self.sam_predictor = None

        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def detect(self, image_path, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8):
        """
        GroundingDINO detection only.
        Returns detections and annotated image.
        """
        image = cv2.imread(image_path)
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Apply NMS
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Annotate
        labels = [f"{classes[class_id]} {conf:0.2f}" for _, _, conf, class_id, _, _ in detections]
        annotated_img = self.box_annotator.annotate(scene=image.copy(), detections=detections)
        for det, label in zip(detections, labels):
            x_min, y_min, _, _ = map(int, det[0])
            cv2.putText(annotated_img, label, (x_min, max(y_min - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return detections, annotated_img

    def segment(self, image, detections):
        """
        Run SAM segmentation on given image and detections.
        Returns detections with masks and annotated image.
        """
        if not self.sam_predictor:
            raise ValueError("SAM is not initialized. Provide sam_encoder_version and sam_ckpt in constructor.")

        # Segment
        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result_masks = []
        for box in detections.xyxy:
            masks, scores, _ = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        detections.mask = np.array(result_masks)

        # Annotate with masks + boxes
        annotated_img = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_img = self.box_annotator.annotate(scene=annotated_img, detections=detections)
        return detections, annotated_img

    def detect_and_segment(self, image_path, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8):
        """
        Full detection + segmentation pipeline.
        Returns detections and annotated image.
        """
        image = cv2.imread(image_path)

        # Step 1: Detect
        detections, _ = self.detect(image_path, classes, box_threshold, text_threshold, nms_threshold)

        # Step 2: Segment
        detections, annotated_img = self.segment(image, detections)
        return detections, annotated_img