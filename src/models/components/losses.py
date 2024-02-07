import segmentation_models_pytorch as smp
import numpy as np
from medpy import metric
import torch.nn.functional as F
from torch import Tensor, nn

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import cv2

from typing import Optional

import torch


class TverskyLoss(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_true, alpha=0.7, beta=0.3):
        y_pred = torch.softmax(y_pred, dim=1)
        y_true = self._one_hot_encoder(y_true)
        loss = 0
        for i in range(1, self.classes):
            p0 = y_pred[:, i, :, :]
            ones = torch.ones_like(p0)
            # p1: prob that the pixel is of class 0
            p1 = ones - p0
            g0 = y_true[:, i, :, :]
            g1 = ones - g0
            # terms in the Tversky loss function combined with weights
            tp = torch.sum(p0 * g0)
            fp = alpha * torch.sum(p0 * g1)
            fn = beta * torch.sum(p1 * g0)
            # add to the denominator a small epsilon to prevent the value from being undefined
            EPS = 1e-5
            num = tp
            den = tp + fp + fn + EPS
            result = num / den
            loss += result
        return 1 - loss / self.classes


class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes=1) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]):  # batch size
            # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(
                        posmask, mode="inner"
                    ).astype(np.uint8)
                    sdf = (negdis - np.min(negdis)) / (
                        np.max(negdis) - np.min(negdis)
                    ) - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                    sdf[boundary == 1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        # outputs_soft = F.softmax(outputs, dim=1)
        outputs_soft = outputs.sigmoid()
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:, self.idx, ...]
        dc = torch.from_numpy(gt_sdf[:, self.idx, ...]).cuda()
        multipled = torch.einsum("bxyz, bxyz->bxyz", pc, dc)
        bd_loss = multipled.mean()
        return bd_loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def _adaptive_size(target):
    target = torch.from_numpy(target).float()
    kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    padding_out = torch.zeros(
        (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
    )
    padding_out[:, 1:-1, 1:-1] = target
    h, w = 3, 3

    Y = torch.zeros(
        (
            padding_out.shape[0],
            padding_out.shape[1] - h + 1,
            padding_out.shape[2] - w + 1,
        )
    )
    for i in range(Y.shape[0]):
        Y[i, :, :] = torch.conv2d(
            target[i].unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1,
        )

    Y = Y * target
    Y[Y == 5] = 0
    C = torch.count_nonzero(Y)
    S = torch.count_nonzero(target)
    smooth = 1e-5
    return (C + smooth) / (S + smooth).item()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    smooth = 1e-5
    boundary_IOU = 0
    for i in range(pred.squeeze().shape[0]):
        pred_boundary = mask_to_boundary(np.uint8(pred[i].squeeze()))
        gt_boundary = mask_to_boundary(np.uint8(gt[i].squeeze()))
        boundary_inter = np.sum(pred_boundary * gt_boundary)
        boundary_union = np.sum(pred_boundary + gt_boundary) - boundary_inter
        boundary_IOU += (
            (boundary_inter + smooth)
            / (boundary_union + smooth)
            / pred.squeeze().shape[0]
        )
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95, boundary_IOU
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, boundary_IOU
    else:
        return 0, 0, boundary_IOU


def get_map(mask):
    if len(np.unique(mask)) <= 3:
        mapping = {0: 0, 1: 128, 2: 255}
    else:
        mapping = {
            0: 0,
            1: 85,
            # 128: 2,
            2: 170,
            3: 255,
        }
    for k in mapping:
        mask[mask == k] = mapping[k]
    return mask


# def test_single_volume(
#     image,
#     label,
#     net,
#     classes,
#     patch_size=[256, 256],
#     test_save_path=None,
#     case=None,
#     z_spacing=1,
# ):
#     image, label = (
#         image.squeeze(0).cpu().detach().numpy(),
#         label.squeeze(0).cpu().detach().numpy(),
#     )
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(
#                     slice, (patch_size[0] / x, patch_size[1] / y), order=3
#                 )  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     big_list = []
#     small_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#         if _adaptive_size(label == i) >= 0.2:
#             small_list.append(metric_list[-1][0])
#         else:
#             big_list.append(metric_list[-1][0])
#
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + "/" + case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + "/" + case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + "/" + case + "_gt.nii.gz")
#     return metric_list, big_list, small_list


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes=1):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).half()
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
        )
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - h + 1,
                padding_out.shape[2] - w + 1,
            )
        ).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0).cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(
            alpha, 0.8
        )  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return loss

    def forward(self, inputs, target):
        # inputs = torch.softmax(inputs, dim=1)
        inputs = inputs.sigmoid()
        # target = self._one_hot_encoder(target)

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        # return self._adaptive_size(inputs, target)
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


class BoundaryDoULossV2(nn.Module):
    def __init__(self, n_classes=1, allowed_outlier_fraction=0.25):
        super(BoundaryDoULossV2, self).__init__()
        self.n_classes = n_classes
        self.allowed_outlier_fraction = allowed_outlier_fraction

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).half()
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
        )
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - h + 1,
                padding_out.shape[2] - w + 1,
            )
        ).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0).cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(
            alpha, 0.8
        )  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return loss

    def forward(self, inputs, target):
        inputs = inputs.sigmoid()

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])

        # Apply outlier fraction logic to BoundaryDoULoss
        output = inputs[:, 0]  # Assuming binary classification

        output = output.float()
        target = target.float()

        pos_mask = target.eq(1.0)
        neg_mask = ~pos_mask

        pt = output.sigmoid().clamp(1e-6, 1 - 1e-6)

        neg_loss = (
            -torch.pow(pt, 2) * torch.nn.functional.logsigmoid(-output) * neg_mask
        )

        if self.allowed_outlier_fraction < 1:
            neg_loss = neg_loss.flatten()
            M = neg_loss.numel()
            num_elements_to_keep = int(M * (1 - self.allowed_outlier_fraction))
            neg_loss, _ = torch.topk(
                neg_loss, k=num_elements_to_keep, largest=False, sorted=False
            )

        neg_loss_reduced = neg_loss.sum() / (neg_mask.sum() + 1e-6)
        loss_outlier = neg_loss_reduced

        return (loss + loss_outlier) / (self.n_classes + 1)


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.bce(y_true, y_pred) + self.dice(y_true, y_pred)


class RegularizedDiceLoss(nn.Module):
    def __init__(self, thresholds):
        super(RegularizedDiceLoss, self).__init__()
        self.thresholds = torch.linspace(0, 1, thresholds + 1)[1:]
        self.target_threshold = self.thresholds[
            torch.argmin(torch.abs(self.thresholds - 0.5))
        ]

    def soft_treshold(self, x, threshold, temperature=10):
        return torch.nn.functional.sigmoid((x - threshold) * temperature)

    def forward(self, logits, targets):
        probs = logits.sigmoid()

        thresholds = self.thresholds.to(probs.device)

        probs_hard = self.soft_treshold(
            probs.flatten().unsqueeze(1), thresholds
        )  # [N T]
        targets = targets.flatten().unsqueeze(1)
        metric = self.fbeta_score(probs_hard, targets, beta=0.5)

        soft_logits = torch.nn.functional.gumbel_softmax(
            metric, tau=1, hard=True, dim=0
        )
        threshold = torch.sum(soft_logits * thresholds, dim=0)
        loss = torch.abs(threshold - self.target_threshold)
        return loss

    def fbeta_score(self, preds, targets, beta=0.5, smooth=1e-5):
        """

        :param preds: N, T
        :param targets: N, 1
        :param beta:
        :param smooth:
        :return:
        """
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)

        tp_fp = tp + fp
        tp_fn = tp + fn

        p = tp / (tp_fp).clamp_min(smooth)
        r = tp / (tp_fn).clamp_min(smooth)

        beta2 = beta**2
        return (1 + beta2) * p * r / (beta2 * p + r).clamp_min(smooth)


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftCLDice(nn.Module):
    def __init__(self, iter_=3, smooth=1.0):
        super(SoftCLDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (
            torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (
            torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (
        torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth
    )
    return 1.0 - coeff


class SoftDiceCLDice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.0):
        super(SoftDiceCLDice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (
            torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (
            torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


class SoftDiceCLDiceBoundaryDoULoss(nn.Module):
    def __init__(self):
        super(SoftDiceCLDiceBoundaryDoULoss, self).__init__()
        self.cldice = SoftDiceCLDice()
        self.dou = BoundaryDoULoss()

    def forward(self, y_true, y_pred):
        return self.cldice(y_true, y_pred) + self.dou(y_pred, y_true)


class AssymetricBinaryFocalLoss(nn.Module):
    __constants__ = ["alpha", "gamma", "reduction", "ignore_index"]

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 2.0,
        allowed_outlier_fraction: float = 0.25,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        eps: float = 1e-6,
        apply_regularisation: bool = False,
        reg_loss_weight: float = 0.01,
    ):
        """

        :param alpha: Prior probability of having positive value in target.
        :param beta: Power factor for dampening weight (focal strength).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.

        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.keep_fraction = 1.0 - allowed_outlier_fraction
        self.eps = eps
        self.apply_regularisation = (
            apply_regularisation and allowed_outlier_fraction > 0
        )
        self.reg_loss_weight = reg_loss_weight

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Compute focal loss for binary classification problem.
        Args:
            inputs: [B,C,H,W]
            targets: [B,C,H,W]

        Returns:

        """

        output = output.float()
        target = target.float()

        pos_mask = target.eq(1.0)

        neg_mask = ~pos_mask

        # ign_mask = gt.eq(self.ignore_index)

        pt = output.sigmoid().clamp(self.eps, 1 - self.eps)

        pos_loss: Tensor = (
            -torch.pow(1 - pt, self.alpha)
            * torch.nn.functional.logsigmoid(output)
            * pos_mask
        )
        neg_loss: Tensor = (
            -torch.pow(pt, self.beta)
            * torch.nn.functional.logsigmoid(-output)
            * neg_mask
        )

        if self.keep_fraction < 1:
            neg_loss = neg_loss.flatten()
            M = neg_loss.numel()
            num_elements_to_keep = int(M * self.keep_fraction)
            neg_loss, _ = torch.topk(
                neg_loss, k=num_elements_to_keep, largest=False, sorted=False
            )

        pos_loss_reduced = pos_loss.sum() / (pos_mask.sum() + self.eps)
        neg_loss_reduced = neg_loss.sum() / (neg_mask.sum() + self.eps)
        loss = pos_loss_reduced + neg_loss_reduced

        if self.apply_regularisation:
            num_outliers = M - num_elements_to_keep
            _, reg_indexes = torch.topk(
                neg_loss, k=num_outliers, largest=True, sorted=False
            )
            logits = torch.flatten(output)[reg_indexes]
            reg_loss = self.binary_shannon_entropy_loss(logits).mean()
            loss = loss + reg_loss * self.reg_loss_weight

        return loss

    def binary_shannon_entropy_loss(self, x):
        pt = x.sigmoid()
        entropy = torch.nn.functional.logsigmoid(
            x
        ) * pt + torch.nn.functional.logsigmoid(-x) * (1 - pt)
        return -entropy


class AssymetricBinaryFocalLossBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = AssymetricBinaryFocalLoss()
        self.boundary = BoundaryDoULoss()

    def forward(self, output: Tensor, target: Tensor):
        return self.focal(output, target) + self.boundary(output, target)


class DTMBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = torch.nn.MSELoss()
        self.boundary = BoundaryDoULoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, output: Tensor, dtm: Tensor, target: Tensor):
        return (
            self.l2(output[:, 1, :, :], dtm[:, 0, :, :])
            + self.boundary(output[:, 0, :, :], target[:, 0, :, :])
            + self.dice(output[:, 0, :, :], target[:, 0, :, :])
        )


class BoundaryDoULoss3D(nn.Module):
    def __init__(self, n_classes=1):
        super(BoundaryDoULoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = (
            torch.Tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )
            .to(target.device)
            .half()
        )

        padding_out = torch.zeros(
            (
                target.shape[0],
                target.shape[-3] + 2,
                target.shape[-2] + 2,
                target.shape[-1] + 2,
            )
        )
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        d, h, w = 3, 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - d + 1,
                padding_out.shape[2] - h + 1,
                padding_out.shape[3] - w + 1,
            )
        ).to(
            target.device
        )  # .cuda()

        for i in range(Y.shape[0]):
            Y[i, :, :, :] = torch.nn.functional.conv3d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0),  # .cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 7] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  # We recommend using a truncated alpha of 0.8.

        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return loss

    def forward(self, inputs, target):
        inputs = inputs.sigmoid()

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        # return self._adaptive_size(inputs, target)
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


# if __name__ == "__main__":
#     import cv2
#
#     def compute_distance_transform(mask):
#         mask = (mask * 255).astype(np.uint8)
#         distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
#         # Normalize distance transform to [0, 1]
#         distance_transform = distance_transform / (distance_transform.max() + 1)
#         return distance_transform
#
#     # # preds = torch.tensor([-5, 5, -5, 5]).float()
#     # # targets = torch.tensor([0, 1, 0, 1]).float()
#     # preds = torch.randn(3, 1, 256, 256).cuda()
#     # targets = torch.randn(3, 1, 256, 256).cuda()
#     # dtm = torch.randn(3, 1, 256, 256).cuda()
#     # # loss = SoftDiceCLDiceBoundaryDoULoss()(preds, targets)
#     # # loss = GenSurfLoss(100)(preds, targets)
#     # loss = GenSurfLoss()(preds, targets, dtm, 0.1)
#     # print(loss)
#
#     x = np.zeros((4, 1, 512, 512))
#     x[:, :, 100:200, 100:200] = 1
#
#     y = np.zeros((4, 1, 512, 512))
#     y[:, :, 100:200, 100:200] = 1
#
#     dtm = torch.from_numpy(np.expand_dims(np.array([compute_distance_transform(s.squeeze()) for s in x]), 1)).cuda()
#
#     x = torch.from_numpy(x).cuda()
#     y = torch.from_numpy(y).cuda()

# loss = GenSurfLoss()(x, y, dtm, 0.1)
# print(loss)


class UncertaintyEstimationLoss(nn.Module):
    def __init__(self, num_samples=10):
        super(UncertaintyEstimationLoss, self).__init__()
        self.base_criterion = BoundaryDoULoss()
        self.num_samples = num_samples

    def forward(self, outputs, targets):
        # outputs: tensor of shape (batch_size, num_classes, height, width)
        # targets: tensor of shape (batch_size, 1, height, width), binary segmentation mask

        total_loss = 0.0

        for _ in range(self.num_samples):
            # Apply dropout during inference
            sampled_outputs = F.dropout(outputs, training=True)

            # Calculate loss using the base criterion (e.g., binary cross entropy)
            loss = self.base_criterion(sampled_outputs, targets)
            total_loss += loss

        # Compute mean loss over all samples
        mean_loss = total_loss / self.num_samples

        # Compute uncertainty as the variance of the sampled losses
        uncertainty = torch.var(
            torch.stack(
                [
                    self.base_criterion(F.dropout(outputs, training=True), targets)
                    for _ in range(self.num_samples)
                ]
            ),
            dim=0,
        )

        # Total loss is a combination of mean loss and uncertainty
        total_loss = mean_loss + uncertainty

        return total_loss


class UncertaintyEstimationLoss3D(nn.Module):
    def __init__(self, num_samples=10):
        super(UncertaintyEstimationLoss3D, self).__init__()
        self.base_criterion = BoundaryDoULoss3D()
        self.num_samples = num_samples

    def forward(self, outputs, targets):
        # outputs: tensor of shape (batch_size, num_classes, depth, height, width)
        # targets: tensor of shape (batch_size, 1, depth, height, width), binary segmentation mask

        total_loss = 0.0

        for _ in range(self.num_samples):
            # Apply dropout during inference
            sampled_outputs = F.dropout3d(outputs, training=True)

            # Calculate loss using the base criterion (e.g., binary cross entropy)
            loss = self.base_criterion(sampled_outputs, targets)
            total_loss += loss

        # Compute mean loss over all samples
        mean_loss = total_loss / self.num_samples

        # Compute uncertainty as the variance of the sampled losses
        uncertainty = torch.var(
            torch.stack(
                [
                    self.base_criterion(F.dropout3d(outputs, training=True), targets)
                    for _ in range(self.num_samples)
                ]
            ),
            dim=0,
        )

        # Total loss is a combination of mean loss and uncertainty
        total_loss = mean_loss + uncertainty

        return total_loss


class StructureLoss(nn.Module):
    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class StructureLossBoundaryDOU(nn.Module):
    def __init__(self):
        super(StructureLossBoundaryDOU, self).__init__()
        self.boundary_loss = BoundaryDoULoss()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean() + self.boundary_loss(pred, mask)


class StructureLossBoundaryDOUV2(nn.Module):
    def __init__(self):
        super(StructureLossBoundaryDOUV2, self).__init__()
        self.boundary_loss = BoundaryDoULoss()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return wiou.mean() + self.boundary_loss(pred, mask)


class WeightedDiceBoundaryDOULoss(nn.Module):
    def __init__(self, edge_weight=1.0):
        super(WeightedDiceBoundaryDOULoss, self).__init__()
        self.edge_weight = edge_weight
        # Define a simple edge detection kernel
        self.edge_kernel = torch.tensor(
            [[[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]],
            dtype=torch.float32,
        ).half()

        self.boundary = BoundaryDoULoss()

    def forward(self, inputs, targets):
        # Standard Dice loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = torch.dot(inputs_flat, targets_flat)
        dice_loss = 1 - (2 * intersection + 1) / (
            inputs_flat.sum() + targets_flat.sum() + 1
        )

        # Edge-enhanced loss
        edge_weights = self.get_edge_weights(targets)
        weighted_intersection = torch.dot(
            inputs_flat * edge_weights.view(-1), targets_flat
        )
        weighted_dice_loss = 1 - (2 * weighted_intersection + 1) / (
            inputs_flat.sum() + targets_flat.sum() + 1
        )

        return dice_loss + weighted_dice_loss + self.boundary(inputs, targets)

    def get_edge_weights(self, masks):
        # Ensure the kernel is on the same device as the masks
        edge_kernel = self.edge_kernel.to(masks.device)
        edge_kernel = edge_kernel.repeat(masks.size(1), 1, 1, 1)

        # Apply edge detection kernel
        edges = F.conv2d(masks, edge_kernel, padding=1)
        edges = edges.clamp(min=0)  # Remove negative values

        # Scale edges to increase their weight
        edge_weights = 1 + edges * self.edge_weight

        return edge_weights


def get_loss(name):
    if name == "StructureLoss":
        return StructureLoss()
    elif name == "BoundaryLoss":
        return BoundaryLoss()
    elif name == "StructureLossBoundaryDOUV2":
        return StructureLossBoundaryDOUV2()
    elif name == "BoundaryDoULoss3D":
        return BoundaryDoULoss3D
    elif name == "UncertaintyEstimationLoss":
        return UncertaintyEstimationLoss()
    elif name == "BoundaryDoULoss":
        return BoundaryDoULoss()
    elif name == "BoundaryDoULoss3D":
        return BoundaryDoULoss3D()
    elif name == "WeightedDiceBoundaryDOULoss":
        return WeightedDiceBoundaryDOULoss()
    else:
        NotImplemented


if __name__ == "__main__":
    loss_bd = BoundaryDoULoss3D().cuda()

    x = torch.zeros((4, 1, 128, 128, 128)).cuda()
    y = torch.rand((4, 1, 128, 128, 128)).cuda()

    print(loss_bd(x, y))
