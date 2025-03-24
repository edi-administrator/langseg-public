import torch
from source.models.lseg_net import OptimizedEmbeddings, LSegOptimizedInference
from typing import Tuple
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def load_model(inference = False, size = [360,480]) -> Tuple[OptimizedEmbeddings | LSegOptimizedInference, Compose]:
    print("Loading model...")

    assert all(x % 32 < 16 for x in size)

    torch.manual_seed(1)
    backbone = "clip_vitl16_384"
    weights_lseg = "src/langseg/weights/lseg_dict_v2.pt"
    weights_query_model = "src/langseg/weights/trained_b12_lre-3_e3_full.pt"

    if inference == False:
        model = OptimizedEmbeddings(
            n_classes = 25,
            embedding_dim = 512
            )
    else:
        model = LSegOptimizedInference(
            backbone=backbone,
            features=256,
            weights_lseg=weights_lseg,
            weigths_query_model=weights_query_model,
            n_classes=25,
            embedding_dim=512,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    transform = Compose(
        [
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Resize(size),
        ]
    )

    print("Model Loaded Successfully!")

    return model, transform
