import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from copy import deepcopy


def _get_backbone_features(model_name, pretrained):
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=3,
        num_classes=0,
        global_pool='',
    )
    if hasattr(backbone, 'num_features'):
        in_features = backbone.num_features
    else:
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = backbone(dummy)
        in_features = out.shape[1]
    return backbone, in_features


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return (self.__class__.__name__
                + '(p=' + '{:.4f}'.format(self.p.data.tolist()[0])
                + ', eps=' + str(self.eps) + ')')


class BirdModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(BirdModel, self).__init__()
        self.backbone, in_features = _get_backbone_features(model_name, pretrained)
        self.global_pool = GeM()
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)[:, :, 0, 0]
        return self.head(pooled)

class AttBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.att = nn.Linear(n_in, n_out)
        self.cla = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor):
        att = torch.sigmoid(self.att(x))
        cla = self.cla(x)
        clip_logits = (att * cla).sum(dim=1) / (att.sum(dim=1) + 1e-6)
        return clip_logits, cla

class BirdSEDModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        use_gru: bool = True,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, in_features = _get_backbone_features(model_name, pretrained)
        self.use_gru = use_gru

        if use_gru:
            self.gru = nn.GRU(
                input_size=in_features,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=gru_dropout if gru_layers > 1 else 0.0,
            )
            att_in = gru_hidden * 2
        else:
            att_in = in_features

        self.dropout = nn.Dropout(p=0.2)
        self.att_block = AttBlock(att_in, num_classes)

    def _encode(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = feats.mean(dim=2)
        feats = feats.permute(0, 2, 1)

        if self.use_gru:
            feats, _ = self.gru(feats)

        feats = self.dropout(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode(x)
        clip_logits, _ = self.att_block(feats)
        return clip_logits

    def forward_frames(self, x: torch.Tensor):
        feats = self._encode(x)
        clip_logits, frame_logits = self.att_block(feats)
        return clip_logits, frame_logits

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)