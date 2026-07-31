import abc
import logging
import torch.nn as nn
import torchvision.transforms.functional as TF

from ..backbone.pvt_v2_eff import pvt_v2_eff_b2
from ProCoGuid.Model.methods import SimpleASPP,MGFF, MAD
from .model_Pipline import SimpleDetector, mask_to_yolo_target
import torchvision.models as models
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from object_classifiy import ImageKMeansCluster
from utils import ops
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


LOGGER = logging.getLogger("main")



def expand_patch_embed_with_class_info(patch_embed, extra_in_channels):
    old_proj = patch_embed.proj
    old_in_channels = old_proj.in_channels
    old_out_channels = old_proj.out_channels

    new_in_channels = old_in_channels + extra_in_channels

    new_proj = nn.Conv2d(new_in_channels, old_out_channels,
                         kernel_size=old_proj.kernel_size,
                         stride=old_proj.stride,
                         padding=old_proj.padding,
                         bias=(old_proj.bias is not None))

    with torch.no_grad():
        # 复制旧权重的前通道
        new_proj.weight[:, :old_in_channels, :, :] = old_proj.weight
        # 初始化新增通道权重
        nn.init.kaiming_normal_(new_proj.weight[:, old_in_channels:, :, :])

        # 复制 bias 数据（如果有）
        if old_proj.bias is not None:
            new_proj.bias.data.copy_(old_proj.bias.data)

    patch_embed.proj = new_proj

def yolo_simple_loss(preds, target, S=7, lambda_coord=5, lambda_noobj=0.5):
    target = target.to(preds.device)
    coord_mask = target[..., 4] > 0
    coord_loss = (preds[..., 0:4] - target[..., 0:4]) ** 2
    coord_loss = coord_loss[coord_mask].sum()
    conf_loss_obj = ((preds[..., 4] - target[..., 4]) ** 2)[coord_mask].sum()
    conf_loss_noobj = ((preds[..., 4] - target[..., 4]) ** 2)[~coord_mask].sum()
    class_loss = ((preds[..., 5] - target[..., 5]) ** 2)[coord_mask].sum()
    loss = lambda_coord * coord_loss + conf_loss_obj + lambda_noobj * conf_loss_noobj + class_loss
    return loss / preds.shape[0]

def xywh2xyxy(box):
    # box: (..., 4)
    x, y, w, h = box.unbind(-1)
    return torch.stack((x - w/2, y - h/2, x + w/2, y + h/2), dim=-1)

bce_loss = nn.BCEWithLogitsLoss(reduction='none')


def decode_predictions(pred, S=7, conf_thresh=0.0, img_size=224,crop_zoom = 0.9):
    B = pred.shape[0]
    cell_size = 1.0 / S
    all_boxes = []

    square_len = img_size * crop_zoom

    for b in range(B):
        best_box = None
        best_conf = -1
        center_x, center_y = 0, 0

        for i in range(S):
            for j in range(S):
                px = pred[b, i, j]
                conf = px[4].item()

                if conf < conf_thresh:
                    continue

                w = max(min(px[2].item(), 0.9), 0.01)
                h = max(min(px[3].item(), 0.9), 0.01)

                cx = min(max((j + px[0].item()) * cell_size, 0.0), 1.0)
                cy = min(max((i + px[1].item()) * cell_size, 0.0), 1.0)

                x1 = (cx - w / 2) * img_size
                y1 = (cy - h / 2) * img_size
                x2 = (cx + w / 2) * img_size
                y2 = (cy + h / 2) * img_size

                x1 = max(0, min(img_size - 1, x1))
                y1 = max(0, min(img_size - 1, y1))
                x2 = max(1, min(img_size, x2))
                y2 = max(1, min(img_size, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = [x1, y1, x2, y2, conf]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

        if best_box is not None:
            half_len = square_len / 2
            sq_x1 = center_x - half_len
            sq_y1 = center_y - half_len
            sq_x2 = center_x + half_len
            sq_y2 = center_y + half_len

            if sq_x1 < 0:
                shift = -sq_x1
                sq_x1 = 0
                sq_x2 += shift
            if sq_y1 < 0:
                shift = -sq_y1
                sq_y1 = 0
                sq_y2 += shift
            if sq_x2 > img_size:
                shift = sq_x2 - img_size
                sq_x2 = img_size
                sq_x1 -= shift
            if sq_y2 > img_size:
                shift = sq_y2 - img_size
                sq_y2 = img_size
                sq_y1 -= shift

            sq_x1 = max(0, min(img_size - 1, sq_x1))
            sq_y1 = max(0, min(img_size - 1, sq_y1))
            sq_x2 = max(1, min(img_size, sq_x2))
            sq_y2 = max(1, min(img_size, sq_y2))

            all_boxes.append([[sq_x1, sq_y1, sq_x2, sq_y2, best_conf]])
        else:
            all_boxes.append([[0, 0, img_size, img_size, 0.0]])

    return all_boxes


import torch.nn.functional as F

import torch
import numpy as np

def crop_and_resize_batch(images, boxes, scales=(0.5, 1.0, 1.5), base_h=384, base_w=384, model_input_size=384):

    B, C, H, W = images.shape
    assert H == W
    scale_factor = H / model_input_size
    scale_results = [[] for _ in scales]
    for i in range(B):
        img = images[i]  # (C, H, W)
        if len(boxes[i]) > 0:
            box = [int(x * scale_factor) for x in boxes[i][0][:4]]
        else:
            box = [0, 0, W, H]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            print(f"[Image {i}] Invalid box after scaling: ({x1},{y1},{x2},{y2}), using full image.")
            x1, y1, x2, y2 = 0, 0, W, H
        cropped = img[:, y1:y2, x1:x2]  # (C, h, w)
        image_np = cropped.permute(1, 2, 0).mul(255).byte().cpu().numpy()  # (h, w, 3)

        resized_images = ops.ms_resize(image_np, scales=scales, base_h=base_h, base_w=base_w)

        for j, resized_np in enumerate(resized_images):
            tensor = torch.from_numpy(resized_np).float().div(255.0).permute(2, 0, 1)  # (3, H, W)
            scale_results[j].append(tensor)

    scale_tensors = [torch.stack(imgs) for imgs in scale_results]
    return scale_tensors


def visualize_crops_with_boxes(images, crops, boxes, save_dir="vis_crops", prefix="img"):
    import time
    os.makedirs(save_dir, exist_ok=True)
    B = images.size(0)

    for i in range(B):
        img_tensor = images[i].cpu()
        crop_tensor = crops[i].cpu()
        box = boxes[i][0] if len(boxes[i]) > 0 else None

        img_pil = TF.to_pil_image(img_tensor)
        crop_pil = TF.to_pil_image(crop_tensor)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(img_pil)
        axs[0].set_title("Original with Box")
        axs[0].axis('off')

        if box:
            x1, y1, x2, y2, conf = box
            x1, y1, x2, y2 = [int(b) for b in [x1, y1, x2, y2]]

            img_w, img_h = images.shape[-1], images.shape[-2]
            if x1 == 0 and y1 == 0 and x2 == img_w and y2 == img_h:
                edge_color = 'blue'
                print(f"[Image {i}] Using fallback whole image box for visualization.")
            else:
                edge_color = 'red'

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=edge_color, facecolor='none')
            axs[0].add_patch(rect)

        axs[1].imshow(crop_pil)
        axs[1].set_title("Cropped & Resized")
        axs[1].axis('off')

        timestamp = int(time.time() * 1000)
        rand_id = torch.randint(0, 1000000, (1,)).item()
        save_path = os.path.join(save_dir, f"{prefix}_{i}_{timestamp}_{rand_id}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)


class Loss(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)

        if self.training:
            mask = data["mask"]
            prob = logits.sigmoid()

            losses = []
            loss_str = []

            sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")
            detection_result = data["detection_result"]
            target = mask_to_yolo_target(mask, S=12, img_size=384, class_idx=0)
            # preds预测得到的结果，target标签
            loss_detection = yolo_simple_loss(detection_result, target)
            losses.append(loss_detection)
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
