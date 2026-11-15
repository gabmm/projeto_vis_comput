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
    - ID 4: Rede 1 (ResNet18) - Estratégia 1 (Fase A: sem congelamento | Fase B: Backbone congelado).
    - ID 5: Rede 1 (ResNet18) - Estratégia 2 (Fase A: sem congelamento | Fase B: Fine-tuning).
    - ID 6: Rede 2 (MobileNetV2) - Estratégia 1 (Fase A: sem congelamento | Fase B: Backbone congelado).
    - ID 7: Rede 2 (MobileNetV2) - Estratégia 2 (Fase A: sem congelamento | Fase B: Fine-tuning).

INPUTS (Parâmetros da função fábrica `get_transfer_model`):
    - exp_id (int): Identificador do experimento (valores aceitos: 4, 5, 6 ou 7).
    - num_classes (int): Número de classes de saída (padrão: 10).

OUTPUTS (Retorno da função fábrica):
    - model (torch.nn.Module): Modelo pré-treinado adaptado para 10 classes.
    - exp_name (str): Nome descritivo do experimento (ex: 'ID4_ResNet18_Stub').
================================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def _build_resnet18(num_classes: int = 10) -> nn.Module:
    """Carrega a ResNet18 pré-treinada na ImageNet e adapta a saída para 10 classes."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Substitui a camada final (original: 512 -> 1000) para (512 -> 10)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v2(num_classes: int = 10) -> nn.Module:
    """Carrega a MobileNetV2 pré-treinada na ImageNet e adapta a saída para 10 classes."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # Substitui a camada final do classificador (original: 1280 -> 1000) para (1280 -> 10)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_transfer_model(exp_id: int, num_classes: int = 10) -> Tuple[nn.Module, str]:
    """
    Fábrica de modelos da Parte 2 (IDs 4 a 7).
    Na Fase A, carrega as duas redes prontas com pesos da ImageNet sem congelar camadas.
    """
    # --------------------------------------------------------------------------
    # REDE PRONTA 1: ResNet18 (IDs 4 e 5)
    # --------------------------------------------------------------------------
    if exp_id == 4:
        model = _build_resnet18(num_classes=num_classes)
        # TODO Estratégia 1: Congelar o backbone convolucional e treinar só `model.fc`
        # Exemplo para Fase B:
        # for name, param in model.named_parameters():
        #     if not name.startswith("fc"):
        #         param.requires_grad = False
        return model, "ID4_ResNet18_Transfer1_Stub"

    elif exp_id == 5:
        model = _build_resnet18(num_classes=num_classes)
        # TODO Estratégia 2: Descongelar últimos blocos (ex: layer4) ou Fine-Tuning total
        return model, "ID5_ResNet18_Transfer2_Stub"

    # --------------------------------------------------------------------------
    # REDE PRONTA 2: MobileNetV2 (IDs 6 e 7)
    # --------------------------------------------------------------------------
    elif exp_id == 6:
        model = _build_mobilenet_v2(num_classes=num_classes)
        # TODO Estratégia 1: Congelar `model.features` e treinar só `model.classifier`
        # Exemplo para Fase B:
        # for param in model.features.parameters():
        #     param.requires_grad = False
        return model, "ID6_MobileNetV2_Transfer1_Stub"

    elif exp_id == 7:
        model = _build_mobilenet_v2(num_classes=num_classes)
        # TODO Estratégia 2: Descongelar parcial ou Fine-Tuning completo
        return model, "ID7_MobileNetV2_Transfer2_Stub"

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