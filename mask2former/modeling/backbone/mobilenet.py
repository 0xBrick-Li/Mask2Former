# Copyright (c) Facebook, Inc. and its affiliates.
import timm

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class D2MobileNetV3Small(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        model_name = cfg.MODEL.MOBILENET.MODEL_NAME
        pretrained = cfg.MODEL.MOBILENET.PRETRAINED
        self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES

        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
        )

        feature_channels = self.backbone.feature_info.channels()
        self._all_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            name: ch for name, ch in zip(self._all_features, feature_channels)
        }

    def forward(self, x):
        feats = self.backbone(x)
        outputs = {}
        for idx, name in enumerate(self._all_features):
            if name in self._out_features:
                outputs[name] = feats[idx]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
