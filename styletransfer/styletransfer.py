def get_loss(with_style_loss=False):
    # TODO: Get content feature from ground truth image and stylized output
    content_feat = net.module.get_content_feat(ray_batch['rgb'].transpose(1, 0).reshape(3, 67, 81).unsqueeze(0))
    output_content_feat = net.module.get_content_feat(ret['rgb'].transpose(1, 0).reshape(3, 67, 81).unsqueeze(0))
    output_style_feats, output_style_feat_mean_std = net.module.get_style_feat(
        ret['rgb'].transpose(1, 0).reshape(3, 67, 81).unsqueeze(0))
    style_feats, style_feat_mean_std = net.module.get_style_feat(style_img.cuda().unsqueeze(0))

    content_loss = get_content_loss(content_feat, output_content_feat)
    style_loss = get_style_loss(style_feat_mean_std, output_style_feat_mean_std)

    if not with_style_loss:
        loss = content_loss
    else:
        loss = content_loss + style_loss

    # Add the loss of first stage
    loss += rgb_loss