import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MergeBlock(nn.Module):
    def __init__(self, encoder, classifier):
        super(MergeBlock, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        encoder_output = self.encoder(x)[-1] # encoder의 forward 결과값은 리스트임, 마지막 레이어의 결과값만 필요하므로
        classifier_output = self.classifier(encoder_output)
        return classifier_output

class HgNet(nn.Module):
     def __init__(self, output_classes=4):
         super(HgNet, self).__init__()
         model = smp.Unet(classes=output_classes, aux_params={"classes": output_classes, "pooling": 'avg', "dropout": 0.2, "activation": "softmax"})
         encoder = model.encoder
         for param in encoder.parameters():  # no params update for encoder layers
             param.requires_grad = False
         classifier = model.classification_head
         model = MergeBlock(encoder, classifier)
         self.model = model

if __name__ == "__main__":
    hgnet = HgNet()
    print(hgnet.model)
