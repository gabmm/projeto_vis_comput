"""
================================================================================
ARQUIVO: src/models_transfer.py
================================================================================
PAPEL NO PROJETO:
    Contém a configuração das redes da Etapa 2 (Transfer Learning). Carrega pelo
    menos duas arquiteturas da literatura com pesos pré-treinados na ImageNet,
    adapta a camada classificadora final para as 10 classes do CIFAR-10 e aplica
    as estratégias de congelamento/descongelamento de camadas (requires_grad).

MAPEAMENTO DE IDs DESTE MÓDULO:
    - ID 4: Rede 1 (ResNet18) - Estratégia 1 (Extração de características: backbone congelado, só a `fc` treina).
    - ID 5: Rede 1 (ResNet18) - Estratégia 2 (Fine-tuning completo: todas as camadas treinam, com lr reduzido no `run_project.py`).
    - ID 6: Rede 2 (MobileNetV2) - Estratégia 1 (Fase A: sem congelamento | Fase B: Backbone congelado).
    - ID 7: Rede 2 (MobileNetV2) - Estratégia 2 (Fase A: sem congelamento | Fase B: Fine-tuning).

INPUTS (Parâmetros da função fábrica `get_transfer_model`):
    - exp_id (int): Identificador do experimento (valores aceitos: 4, 5, 6 ou 7).
    - num_classes (int): Número de classes de saída (padrão: 10).

OUTPUTS (Retorno da função fábrica):
    - model (torch.nn.Module): Modelo pré-treinado adaptado para 10 classes.
    - exp_name (str): Nome descritivo do experimento (ex: 'ID4_ResNet18_FeatExtract').
================================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


# Lado, em pixels, para o qual a entrada 32x32 é ampliada (224 = resolução do pré-treino).
INPUT_SIZE = 224


class UpsampledPretrainedNet(nn.Module):
    """
    Adapta uma rede pré-treinada na ImageNet ao CIFAR-10 (3x32x32):
      1. Amplia a entrada para INPUT_SIZE x INPUT_SIZE. A ResNet18 reduz a resolução
         em 32x, então 32x32 chegaria ao último bloco como um mapa 1x1.
      2. Mantém em eval() os BatchNorm congelados: `requires_grad=False` não congela
         a média e a variância correntes, e o `engine.py` chama `model.train()`.
    """
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.upsample = nn.Upsample(size=INPUT_SIZE, mode="bilinear", align_corners=False)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.upsample(x))

    def train(self, mode: bool = True) -> "UpsampledPretrainedNet":
        super().train(mode)
        # Devolve ao modo eval só os BatchNorm congelados; o resto segue `mode`
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) and not module.weight.requires_grad:
                module.eval()
        return self

def _build_resnet18(num_classes: int = 10, freeze_backbone: bool = False) -> nn.Module:
    """
    Carrega a ResNet18 pré-treinada na ImageNet e adapta a saída para 10 classes.
    Com `freeze_backbone=True` (Estratégia 1) só a nova camada `fc` fica treinável;
    com False (Estratégia 2) todos os pesos seguem treináveis.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Congela antes de trocar a `fc`: a camada nova já nasce treinável
        model.requires_grad_(False)

    # Substitui a camada final (original: 512 -> 1000) para (512 -> 10)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return UpsampledPretrainedNet(model)


def _build_mobilenet_v2(num_classes: int = 10, freeze_backbone: bool = False) -> nn.Module:
    """Carrega a MobileNetV2 pré-treinada na ImageNet e adapta a saída para 10 classes."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = not freeze_backbone

    # Substitui a camada final do classificador (original: 1280 -> 1000) para (1280 -> 10)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return UpsampledPretrainedNet(model)


def get_transfer_model(exp_id: int, num_classes: int = 10) -> Tuple[nn.Module, str]:
    """
    Fábrica de modelos da Parte 2 (IDs 4 a 7).
    Os IDs 4 e 5 já aplicam as suas estratégias (backbone congelado e fine-tuning
    completo). Os IDs 6 e 7 ainda carregam a rede com pesos da ImageNet sem
    congelar camadas (Fase A).
    """
    # --------------------------------------------------------------------------
    # REDE PRONTA 1: ResNet18 (IDs 4 e 5)
    # --------------------------------------------------------------------------
    if exp_id == 4:
        # Estratégia 1: backbone congelado, só a camada `fc` (512 -> 10) treina
        model = _build_resnet18(num_classes=num_classes, freeze_backbone=True)
        return model, "ID4_ResNet18_FeatExtract"

    elif exp_id == 5:
        # Estratégia 2: fine-tuning completo, nenhuma camada congelada.
        # Usa lr menor que o global (ver EXP_OVERRIDES no `run_project.py`).
        model = _build_resnet18(num_classes=num_classes, freeze_backbone=False)
        return model, "ID5_ResNet18_FineTuning"

    # --------------------------------------------------------------------------
    # REDE PRONTA 2: MobileNetV2 (IDs 6 e 7)
    # --------------------------------------------------------------------------
    elif exp_id == 6:
        # Backbone congelado
        model = _build_mobilenet_v2(num_classes=num_classes, freeze_backbone=True)
        return model, "ID6_MobileNetV2_FreezeBackbone"

    elif exp_id == 7:
        # Nenhuma camada congelada (lr menor)
        model = _build_mobilenet_v2(num_classes=num_classes, freeze_backbone=False)
        return model, "ID7_MobileNetV2_FineTuning"

    else:
        raise ValueError(
            f"[ERRO] exp_id={exp_id} inválido para models_transfer.py. Use valores entre 4 e 7."
        )


# ==============================================================================
# BLOCO DE AUTOTESTE (Permite testar o arquivo isoladamente no terminal)
# Como rodar: python3 src/models_transfer.py
# ==============================================================================
if __name__ == "__main__":
    print("[TESTE] Validando instâncias de src/models_transfer.py (IDs 4 a 7)...\n")
    dummy_input = torch.randn(4, 3, 32, 32)  # Lote falso de 4 imagens RGB 32x32

    for current_id in range(4, 8):
        model, name = get_transfer_model(exp_id=current_id, num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"-> Experimento: {name} (ID {current_id})")
        print(f"   Shape de Saída: {output.shape} (Esperado: torch.Size([4, 10]))")
        print(f"   Parâmetros Totais: {total_params:,} | Treináveis: {trainable_params:,}\n")