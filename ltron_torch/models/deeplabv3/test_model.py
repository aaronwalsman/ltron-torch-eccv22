from ltron_torch.models.deeplabv3 import deeplabv3

model = deeplabv3.deeplabv3_resnet50(pretrained=False, pretrained_backbone=True, aux_loss=False)
print(model)