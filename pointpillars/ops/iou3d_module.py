# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/iou3d_utils.py

import torch
from pointpillars.ops.iou3d_op import boxes_overlap_bev_gpu, boxes_iou_bev_gpu
# Comment out the CUDA NMS imports
# from pointpillars.ops.iou3d_op import nms_gpu, nms_normal_gpu

# Import CPU NMS from torchvision
from torchvision.ops import nms as cpu_nms


def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate boxes Overlap in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_overlap (torch.Tensor): Overlap result with shape (M, N).
    """
    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)
    return ans_overlap


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)
    return ans_iou


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """CPU fallback for NMS using torchvision.ops.nms.

    Args:
        boxes (torch.Tensor): Input boxes with shape [N, 5] ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with shape [N].
        thresh (float): IoU threshold.
        pre_maxsize (int, optional): Limit on boxes before NMS.
        post_max_size (int, optional): Limit on boxes after NMS.

    Returns:
        torch.Tensor: Indices of boxes kept after NMS.
    """
    # Sort scores in descending order
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    
    # Use only the first 4 coordinates for NMS
    boxes_2d = boxes[order][:, :4].contiguous()
    scores_ordered = scores[order].contiguous()
    
    # Perform CPU NMS using torchvision
    from torchvision.ops import nms as cpu_nms  # import here if not at the top
    keep_indices = cpu_nms(boxes_2d, scores_ordered, thresh)
    
    if post_max_size is not None:
        keep_indices = keep_indices[:post_max_size]
    
    # Map back to original indices
    return order[keep_indices].contiguous()




def nms_normal_gpu(boxes, scores, thresh):
    """CPU fallback for normal NMS using torchvision.ops.nms.

    Args:
        boxes (torch.Tensor): Input boxes with shape [N, 5].
        scores (torch.Tensor): Scores of boxes with shape [N].
        thresh (float): IoU threshold.

    Returns:
        torch.Tensor: Indices of boxes kept after NMS.
    """
    order = scores.sort(0, descending=True)[1]
    boxes_2d = boxes[order][:, :4].contiguous()
    scores = scores[order].contiguous()
    
    keep_indices = cpu_nms(boxes_2d, scores, thresh)
    
    return order[keep_indices].contiguous()
