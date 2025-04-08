#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
import torch
from gtts import gTTS
from playsound import playsound
import pdb

# Import utility functions from pointpillars
from pointpillars.model import PointPillars
from pointpillars.utils import (
    read_points, write_pickle, read_calib, read_label,
    keep_bbox_from_image_range, keep_bbox_from_lidar_range,
    vis_pc, vis_img_3d, bbox3d2corners_camera, points_camera2image,
    bbox_camera2lidar
)

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """
    Filter points that fall within the defined range.
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]

def annotate_image(image, results, classes, limited_distance=None, P2=None, calib_info=None, pcd_limit_range=None, pc=None):
    
    annotated = image.copy()

    # If calibration info is provided, filter detections by image range.
    if calib_info is not None and annotated is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)
        image_shape = annotated.shape[:2]
        # Convert GPU tensors to NumPy arrays if needed.
        if torch.is_tensor(results['lidar_bboxes']):
            results['lidar_bboxes'] = results['lidar_bboxes'].cpu().numpy()
        if 'camera_bboxes' in results and torch.is_tensor(results['camera_bboxes']):
            results['camera_bboxes'] = results['camera_bboxes'].cpu().numpy()
        results = keep_bbox_from_image_range(results, tr_velo_to_cam, r0_rect, P2, image_shape)
    
    # Filter by lidar range if provided.
    if pcd_limit_range is not None:
        results = keep_bbox_from_lidar_range(results, pcd_limit_range)
    
    # Apply limited distance filtering.
    lidar_bboxes = results['lidar_bboxes']
    if limited_distance is not None:
        centers = lidar_bboxes[:, :3]
        dists = np.linalg.norm(centers, axis=1)
        keep = dists <= limited_distance
        results['lidar_bboxes'] = lidar_bboxes[keep]
        results['labels'] = np.array(results['labels'])[keep]
        results['scores'] = np.array(results['scores'])[keep]
        if 'camera_bboxes' in results:
            results['camera_bboxes'] = results['camera_bboxes'][keep]
    
    # If camera boxes are available, use them to draw 2D boxes and compute left/right/center.
    if 'camera_bboxes' in results and results['camera_bboxes'] is not None:
        bboxes_corners = bbox3d2corners_camera(results['camera_bboxes'])
        image_points = points_camera2image(bboxes_corners, P2)
        h, w = annotated.shape[:2]
        position_list = []    # Stores left/right/center.
        distance_list = []    # Stores computed distances.
        
        # Tolerance (10% of image width) for center region.
        tolerance = 0.1 * w

        for i, pts in enumerate(image_points):
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            # Clip coordinates.
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            # cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Determine left/right/center relative to image center.
            bbox_center_x = (x_min + x_max) / 2.0
            image_center_x = w / 2.0
            if bbox_center_x < image_center_x - tolerance:
                pos = "Left"
            elif bbox_center_x > image_center_x + tolerance:
                pos = "Right"
            else:
                pos = "Center"
            position_list.append(pos)
            
            # Compute distance using the lidar box center.
            if torch.is_tensor(results['lidar_bboxes'][i]):
                bbox = results['lidar_bboxes'][i].cpu().numpy()
            else:
                bbox = results['lidar_bboxes'][i]
            center = bbox[:3]
            distance = np.linalg.norm(center)
            distance_list.append(distance)
            
            text = f"{classes.get(results['labels'][i], 'Unknown')} | {distance:.1f}m | {pos}"
            cv2.putText(annotated, text, (x_min, y_max - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        results['positions'] = position_list
        results['distances'] = distance_list
        # Draw the 3D wireframe boxes.
        annotated = vis_img_3d(annotated, image_points, results['labels'], rt=True)

    else:
        # Fallback: if no camera boxes, use approximate lidar bbox.
        h, w = annotated.shape[:2]
        angle_list = []
        distance_list = []
        for i, (label, score, bbox) in enumerate(zip(results['labels'], results['scores'], results['lidar_bboxes'])):
            if torch.is_tensor(bbox):
                bbox = bbox.cpu().numpy()
            x, y, z, dx, dy, dz, rot = bbox
            center = np.array([x, y, z])
            distance = np.linalg.norm(center)
            distance_list.append(distance)
            x_min = int(x - dx/2)
            y_min = int(y - dy/2)
            x_max = int(x + dx/2)
            y_max = int(y + dy/2)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            # cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # In this fallback, we still include an angle (set to 0) since no camera info is available.
            angle_deg = 0.0
            angle_list.append(angle_deg)
            text = f"{classes.get(label, 'Unknown')} | {distance:.1f}m | {angle_deg:.1f} deg"
            cv2.putText(annotated, text, (x_min, y_max - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        results['angle_degs'] = angle_list
        results['distances'] = distance_list
    
    return results, annotated

def generate_audio(results, classes, num_samples=5, audio_filename="announcement.mp3"):
    """
    Generate an audio announcement for the closest detections.
    """
    distances = results['distances']
    if torch.is_tensor(distances):
        distances = distances.cpu().numpy()
    sorted_indices = np.argsort(distances)
    samples = sorted_indices[:num_samples]
    
    announcements = []
    for idx in samples:
        label = results['labels'][idx]
        score = results['scores'][idx]
        bbox = results['lidar_bboxes'][idx]
        if torch.is_tensor(bbox):
            center = bbox[:3].cpu().numpy()
        else:
            center = np.array(bbox)[:3]
        distance = np.linalg.norm(center)
        # Include left/right/center if available.
        if 'positions' in results:
            pos = results['positions'][idx]
            announcement = f"Object {classes.get(label, 'Unknown')} detected at {distance:.1f} meters, position {pos}, confidence {score:.2f}."
        else:
            announcement = f"Object {classes.get(label, 'Unknown')} detected at {distance:.1f} meters with confidence {score:.2f}."
        announcements.append(announcement)
    
    final_text = " ".join(announcements)
    print("\nAudio Announcements:")
    for ann in announcements:
        print(ann)
    
    tts = gTTS(text=final_text, lang='en')
    tts.save(audio_filename)  # Save as MP3.
    playsound(audio_filename)
    os.remove(audio_filename)
    return announcements

def main(args):
    # Define classes.
    CLASSES = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # --- Model Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # model = PointPillars(nclasses=len(CLASSES)).to(device)
    # if not args.no_cuda:
    #     model.load_state_dict(torch.load(args.ckpt))
    # else:
    #     model.load_state_dict(torch.load(args.ckpt, map_location=device))
    # model.eval()

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        if args.finetune:
            checkpoint = torch.load(args.ckpt, weights_only=True)  # Security fix
            model.load_state_dict(checkpoint['model_state_dict'])  # Extract model weights only
        else:
            model.load_state_dict(torch.load(args.ckpt))

    else:
        model = PointPillars(nclasses=len(CLASSES))
        # model.load_state_dict(
        #     torch.load(args.ckpt, map_location=torch.device('cpu')))
        # model = PointPillars(nclasses=len(CLASSES))
        # model.load_state_dict(torch.load(args.ckpt, map_location=device))
        if args.finetune:
            checkpoint = torch.load(args.ckpt, weights_only=True, map_location=torch.device("mps"))  # Security fix
            model.load_state_dict(checkpoint['model_state_dict'])  # Extract model weights only
        else:
            model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        image = cv2.imread(args.img_path, 1)
    else:
        image = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.to(device)
        
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]
    if calib_info is not None and image is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = image.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    results = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = results['lidar_bboxes']
    labels, scores = results['labels'], results['scores']

    # vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    lidar_bboxes = results['lidar_bboxes']
    if not torch.is_tensor(lidar_bboxes):
        lidar_bboxes = torch.tensor(lidar_bboxes, device=device)
    results['lidar_bboxes'] = lidar_bboxes

    # Compute distances (this is used for printing, but annotation will compute distances again).
    locations = lidar_bboxes[:, :3]
    distances = torch.sqrt(torch.sum(locations ** 2, dim=1))
    results['distances'] = distances

    # --- Print Detection Results ---
    distances_np = distances.cpu().numpy()
    print("\nDetection Results with Distances:")
    for i, (label, score, dist) in enumerate(zip(results['labels'], results['scores'], distances_np)):
        if dist < args.limited_distance:
            print(f"Obj {i+1}: {CLASSES.get(label, 'Unknown')} | Conf: {score:.2f} | Dist: {dist:.2f}m")
        else:
            print(f"Obj {i+1}: {CLASSES.get(label, 'Unknown')} | Conf: {score:.2f} | Dist: {dist:.2f}m (out of range)")

    # --- Optional: Annotate Image ---
    if image is not None:
        results, annotated = annotate_image(image, results, CLASSES, 
                                             limited_distance=args.limited_distance, 
                                             P2=P2, calib_info=calib_info, 
                                             pcd_limit_range=pcd_limit_range, pc=pc)
        output_image = "annotated_output.png"
        cv2.startWindowThread()
        cv2.imshow(f'{os.path.basename(args.img_path)}-3d bbox', annotated)
        cv2.imwrite('bbox.png', annotated)
        cv2.waitKey(1) 
        print(f"Annotated image saved as {output_image}")

    # --- Optional: Visualize Point Cloud ---
    try:
        vis_pc(pc, bboxes=results['lidar_bboxes'], labels=results['labels'])
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")

    # --- Optional: Ground Truth Visualization ---
    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]
        # Use -1 label to denote ground truth.
        gt_labels = [-1] * len(gt_label['name'])
        pred_gt_lidar_bboxes = np.concatenate([results['lidar_bboxes'], gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([results['labels'], gt_labels])
        vis_pc(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)

    # --- Optional: Generate Audio ---
    if args.generate_audio:
        path, _ = generate_audio(results, CLASSES, num_samples=5)
        playsound(path)
    
    # --- Optional: Save Results ---
    if args.output_path:
        write_pickle(results, args.output_path)
        print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointPillars Inference with Distance and Annotation')
    parser.add_argument('--ckpt', required=True, help='Path to model checkpoint (e.g., model.pt)')
    parser.add_argument('--pc_path', required=True, help='Path to pointcloud.bin')
    parser.add_argument('--calib_path', default='', help='Path to calibration file (optional)')
    parser.add_argument('--gt_path', default='', help='Path to ground truth label file (optional)')
    parser.add_argument('--img_path', default='', help='Path to image for annotation (optional)')
    parser.add_argument('--output_path', default='', help='Path to save output pickle file (optional)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable GPU and use CPU')
    parser.add_argument('--limited_distance', type=float, default=50, help='Only annotate objects within this distance (meters)')
    parser.add_argument('--generate_audio', action='store_true', help='Generate audio announcements for closest detections')
    parser.add_argument('--finetune', action='store_true', help='Use fine-tuned model')

    args = parser.parse_args()
    
    main(args)
