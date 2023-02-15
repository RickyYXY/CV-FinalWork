import torch
import torch.nn as nn
import torch.nn.functional as F


def seq_photo_loss(flow_preds_fw, flow_preds_bw, im1, im2, gamma=0.8,):
    n_predictions = len(flow_preds_fw)
    photo_loss_sum = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)

        occ_fw, occ_bw = forward_backward_occ_check(
            flow_preds_fw[i], flow_preds_bw[i])
        # warped im1 by forward flow and im2
        im1_warp = torch_warp(im2, flow_preds_fw[i])
        im2_warp = torch_warp(im1, flow_preds_bw[i])

        photo_loss = photo_loss_multi_type(
            im1, im1_warp, occ_fw, photo_loss_use_occ=True)
        photo_loss += photo_loss_multi_type(
            im2, im2_warp, occ_bw, photo_loss_use_occ=True)

        photo_loss_sum += i_weight * photo_loss

    return photo_loss_sum


def photo_loss_multi_type(x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                          photo_loss_delta=0.4, photo_loss_use_occ=False,
                          ):
    occ_weight = occ_mask
    if photo_loss_type == 'abs_robust':
        photo_diff = x - y
        loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
    elif photo_loss_type == 'charbonnier':
        photo_diff = x - y
        loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
    elif photo_loss_type == 'L1':
        photo_diff = x - y
        loss_diff = torch.abs(photo_diff + 1e-6)
    elif photo_loss_type == 'SSIM':
        loss_diff, occ_weight = weighted_ssim(x, y, occ_mask)
    else:
        raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

    if photo_loss_use_occ:
        photo_loss = torch.sum(loss_diff * occ_weight) / \
            (torch.sum(occ_weight) + 1e-6)
    else:
        photo_loss = torch.mean(loss_diff)
    return photo_loss


def forward_backward_occ_check(flow_fw, flow_bw, alpha1=0.1, alpha2=0.5, obj_out_all='obj'):
    """
    In this function, the parameter alpha needs to be improved
    alpha1 in UnFlow is 0.01, may need check
    """

    def length_sq_v0(x):
        # torch.sum(x ** 2, dim=1, keepdim=True)
        # temp = torch.sum(x ** 2, dim=1, keepdim=True)
        # temp = torch.pow(temp, 0.5)
        return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
        # return temp

    def length_sq(x):
        # torch.sum(x ** 2, dim=1, keepdim=True)
        temp = torch.sum(x ** 2, dim=1, keepdim=True)
        temp = torch.pow(temp, 0.5)
        # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
        return temp

    # if self.sum_abs_or_squar:
    #     sum_func = length_sq_v0
    # else:
    #     sum_func = length_sq
    sum_func = length_sq_v0

    mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
    flow_bw_warped = torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
    flow_fw_warped = torch_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh = alpha1 * mag_sq + alpha2
    # 0 means the occlusion region where the photo loss we should ignore
    occ_fw = sum_func(flow_diff_fw) < occ_thresh
    occ_bw = sum_func(flow_diff_bw) < occ_thresh
    occ_fw = occ_fw.float()
    occ_bw = occ_bw.float()
    # if IF_DEBUG:
    #     temp_ = sum_func(flow_diff_fw)
    #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
    #     temp_ = sum_func(flow_diff_bw)
    #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
    #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
    #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
    if obj_out_all == 'obj':
        out_occ_fw = torch_outgoing_occ_check(flow_fw)
        out_occ_bw = torch_outgoing_occ_check(flow_bw)
        occ_fw = torch_get_obj_occ_check(
            occ_mask=occ_fw, out_occ=out_occ_fw)
        occ_bw = torch_get_obj_occ_check(
            occ_mask=occ_bw, out_occ=out_occ_bw)
    return occ_fw, occ_bw


def torch_outgoing_occ_check(flow):

    B, C, H, W = flow.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
    flow_x, flow_y = torch.split(flow, 1, 1)
    if flow.is_cuda:
        xx = xx.cuda()
        yy = yy.cuda()
    # tools.check_tensor(flow_x, 'flow_x')
    # tools.check_tensor(flow_y, 'flow_y')
    # tools.check_tensor(xx, 'xx')
    # tools.check_tensor(yy, 'yy')
    pos_x = xx + flow_x
    pos_y = yy + flow_y
    # tools.check_tensor(pos_x, 'pos_x')
    # tools.check_tensor(pos_y, 'pos_y')
    # print(' ')
    # check mask
    outgoing_mask = torch.ones_like(pos_x)
    outgoing_mask[pos_x > W - 1] = 0
    outgoing_mask[pos_x < 0] = 0
    outgoing_mask[pos_y > H - 1] = 0
    outgoing_mask[pos_y < 0] = 0
    return outgoing_mask.float()


def torch_get_obj_occ_check(occ_mask, out_occ):
    outgoing_mask = torch.zeros_like(occ_mask)
    if occ_mask.is_cuda:
        outgoing_mask = outgoing_mask.cuda()
    outgoing_mask[occ_mask == 1] = 1
    outgoing_mask[out_occ == 0] = 1
    return outgoing_mask


def torch_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # print(grid.shape,flo.shape,'...')
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
    # tools.check_tensor(x, 'x')
    # tools.check_tensor(vgrid, 'vgrid')
    output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
    # mask = torch.autograd.Variable(torch.ones(x.size()))
    # if x.is_cuda:
    #     mask = mask.cuda()
    # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
    #
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1
    # output = output * mask
    # # nchw->>>nhwc
    # if x.is_cuda:
    #     output = output.cpu()
    # output_im = output.numpy()
    # output_im = np.transpose(output_im, (0, 2, 3, 1))
    # output_im = np.squeeze(output_im)
    return output


def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.
    Args:
      x: a batch of images, of shape [B, C, H, W].
      y:  a batch of images, of shape [B, C, H, W].
      weight: shape [B, 1, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations). values are in [0,1]
      c1: A floating point number, regularizes division by zero of the means.
      c2: A floating point number, regularizes division by zero of the second
        moments.
      weight_epsilon: A floating point number, used to regularize division by the
        weight.

    Returns:
      A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
      similarity loss per pixel per channel, and the second, of shape
      [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
      know how much to weigh each pixel in the first tensor. For example, if
      `'weight` was very small in some area of the images, the first tensor will
      still assign a loss to these pixels, but we shouldn't take the result too
      seriously.
    """

    def _avg_pool3x3(x):
        # tf kernel [b,h,w,c]
        return F.avg_pool2d(x, (3, 3), (1, 1))
        # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                         'likely unintended.')
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / \
        (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
    sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x ** 2 + mu_y ** 2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight
