from .ops import ConvBNReLU,PixelNormalizer
import torch.nn as nn
import torch
import timm
import torchvision.transforms.functional as FF
import torch.nn.functional as F

class SimpleDetector(nn.Module):
    def __init__(self, pretrained=True, input_norm=True, S=7, B=1, C=1, encoder = None):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        # self.encoder = encoder

        channel = [2048,1024,512,256,64]
        self.tra_5 = ConvBNReLU(in_planes=channel[0], out_planes=channel[4], kernel_size=3, stride=1, padding=1)

        self.conv_final = nn.ModuleList([
            ConvBNReLU(in_planes=channel[4], out_planes=channel[4], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_planes=channel[4], out_planes=channel[4], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_planes=channel[4], out_planes=channel[4], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_planes=channel[4], out_planes=channel[4], kernel_size=3, stride=1, padding=1),
        ])

        self.encoder = timm.create_model(
            model_name="resnet50", features_only=True, out_indices=range(5), pretrained=False
        )
        if pretrained:
            params = torch.hub.load_state_dict_from_url(
                url="https://github.com/lartpang/Archieve/releases/download/pretrained-model/resnet50-timm.pth",
                map_location="cpu",
            )
            self.encoder.load_state_dict(params, strict=False)


        self.head = nn.Conv2d(64, B * (5 + C), kernel_size=1)  # 输出通道数 = B*(5+C)
        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()

    def normalize_encoder(self, x):
            x = self.normalizer(x)
            c1, c2, c3, c4, c5 = self.encoder(x)
            return c1, c2, c3, c4, c5


    def forward(self, x):
        m_trans_feats = self.normalize_encoder(x)
        m5 = self.tra_5(m_trans_feats[4])
        x = self.head(m5)  # (B, B*(5+C), S, S)
        x = x.permute(0, 2, 3, 1)  # (B, S, S, B*(5+C))
        return x

def decode_predictions(pred, S=7, conf_thresh=0.0, img_size=224):

    B = pred.shape[0]
    cell_size = 1.0 / S
    all_boxes = []

    for b in range(B):
        best_box  = None
        best_conf = -1

        for i in range(S):
            for j in range(S):
                px   = pred[b, i, j]
                conf = px[4].item()

                if conf < conf_thresh:
                    continue

                cx = (j + px[0].item()) * cell_size
                cy = (i + px[1].item()) * cell_size
                w  = px[2].item()
                h  = px[3].item()

                x1 = (cx - w / 2) * img_size
                y1 = (cy - h / 2) * img_size
                x2 = (cx + w / 2) * img_size
                y2 = (cy + h / 2) * img_size

                if conf > best_conf:
                    best_conf = conf
                    best_box  = [x1, y1, x2, y2, conf]

        if best_box is not None:
            all_boxes.append([best_box])
        else:
            all_boxes.append([[0, 0, img_size, img_size, 1.0]])

    return all_boxes

import torch
import torchvision.transforms.functional as FF

def crop_and_resize_batch(images, boxes, target_size=(384, 384)):

    B, C, H, W = images.shape
    result = []

    for i in range(B):
        img = images[i]
        box = boxes[i][0] if len(boxes[i]) > 0 else [0, 0, W, H]  # fallback: whole image
        x1, y1, x2, y2 = [int(x) for x in box[:4]]

        # 限制坐标在图像范围内
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(1, min(W, x2))
        y2 = max(1, min(H, y2))

        if x2 <= x1 or y2 <= y1:  # 防止无效框
            x1, y1, x2, y2 = 0, 0, W, H

        cropped = img[:, y1:y2, x1:x2]
        pil_img = FF.to_pil_image(cropped)
        resized = FF.resize(pil_img, target_size)
        result.append(FF.to_tensor(resized))

    return torch.stack(result)



def ms_resize_tensor(img_tensor, scales, base_h=None, base_w=None, mode='bilinear'):
    assert isinstance(scales, (list, tuple))
    B, C, H, W = img_tensor.shape
    if base_h is None:
        base_h = H
    if base_w is None:
        base_w = W

    outputs = []
    for s in scales:
        new_h = int(base_h * s)
        new_w = int(base_w * s)
        resized = F.interpolate(img_tensor, size=(new_h, new_w), mode=mode, align_corners=False)
        outputs.append(resized)
    return outputs

def mask_to_yolo_target(mask, S=12, img_size=384, class_idx=0):
    B, _, H, W = mask.shape
    target = torch.zeros((B, S, S, 6), dtype=torch.float32)

    cell_size = img_size / S

    for b in range(B):
        m = mask[b, 0]  # (H, W)
        if m.sum() == 0:
            continue
        ys, xs = torch.nonzero(m, as_tuple=True)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # 归一化中心和尺寸
        cx = ((x_min + x_max) / 2) / img_size
        cy = ((y_min + y_max) / 2) / img_size
        w = (x_max - x_min) / img_size
        h = (y_max - y_min) / img_size
        i = int(cy * S)
        j = int(cx * S)
        i = min(i, S - 1)
        j = min(j, S - 1)

        cell_cx = cx * S - j
        cell_cy = cy * S - i

        target[b, i, j, 0] = cell_cx
        target[b, i, j, 1] = cell_cy
        target[b, i, j, 2] = w
        target[b, i, j, 3] = h
        target[b, i, j, 4] = 1.0  # confidence
        target[b, i, j, 5] = float(class_idx)  # class
    return target
