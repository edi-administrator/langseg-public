import torch
import torch.nn as nn
import clip


from .lseg_blocks import (
    Interpolate,
    _make_encoder,
    FeatureFusionBlock_custom,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(torch.nn.Module):
    def __init__(
        self,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        use_bn=False,
    ):
        super().__init__()

        # idk why, check this
        use_bn = True

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=[5, 11, 17, 23],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.scratch.output_conv = head

    def forward(self, x) -> torch.Tensor:
        """
        Returns image embeddings of torch size [1, 512, 176, 240]
        """

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        image_features = self.scratch.head1(path_1)

        return image_features

    def text_embeddings(self, text) -> torch.Tensor:
        """
        Returns text embeddings of torch size [1, 512]
        """

        tokenized_text = clip.tokenize(text).to(torch.device("cuda:0"))
        text_features = self.clip_pretrained.encode_text(tokenized_text)

        return text_features

    def cosine_similarity(self, image_embeddings, text_features) -> torch.Tensor:
        """
        Preprocess embeddings and perform cosine similarity
        """

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        batch_size, height, width, channels = image_embeddings.shape
        num_text_features, _ = text_features.shape

        image_embeddings = image_embeddings.reshape(-1, 512)
        text_features = text_features.permute(1, 0)

        result = (image_embeddings @ text_features).reshape(
            batch_size, height, width, num_text_features
        )

        result = result.permute(0, 3, 1, 2)

        return result


class OptimizedEmbeddings(torch.nn.Module):

    def __init__(self, n_classes: int, embedding_dim: int) -> None:
        super().__init__()

        torch.manual_seed(100)

        self.text_vector = nn.Parameter(
            torch.randn((n_classes, embedding_dim)).to(torch.device("cuda:0")),
            requires_grad=True,
        )

    def forward(self, image_embeddings) -> torch.Tensor:

        text_features = self.text_vector / self.text_vector.norm(dim=-1, keepdim=True)

        # text_features = self.text_vector
        batch_size, height, width, channels = image_embeddings.shape
        num_text_features, _ = text_features.shape

        image_embeddings = image_embeddings.reshape(-1, 512)
        text_features = text_features.permute(1, 0)

        result = (image_embeddings @ text_features).view(
            batch_size, height, width, num_text_features
        )

        result = result.permute(0, 3, 1, 2)

        return result


class LSegOptimizedInference(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        features: int,
        weights_lseg: str,
        weigths_query_model: str,
        n_classes: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.net = LSeg(backbone=backbone, features=features)
        self.net.load_state_dict(torch.load(weights_lseg))

        self.query_model = OptimizedEmbeddings(
            n_classes=n_classes, embedding_dim=embedding_dim
        )
        self.query_model.load_state_dict(torch.load(weigths_query_model))

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0,2,3,1)
        return self.query_model(x)
