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
    - ID 5: Rede 1 (ResNet18) - Estratégia 2 (Fase A: sem congelamento | Fase B: Fine-tuning).
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


# Lado, em pixels, para o qual a entrada 32x32 é ampliada antes da rede pré-treinada.
# 224 é a resolução do pré-treino na ImageNet. Sem GPU, 96 é a alternativa barata
# (ver a sondagem na docstring de `UpsampledPretrainedNet`).
INPUT_SIZE = 224


class UpsampledPretrainedNet(nn.Module):
    """
    Envolve uma rede pré-treinada na ImageNet para recebê-la no CIFAR-10 (3x32x32).
    Usada pela Rede 1 (ResNet18, IDs 4 e 5). Não altera nenhum peso da rede
    original; acrescenta apenas dois cuidados em volta dela:

      1. Ampliação da entrada: Upsample bilinear de 32x32 para INPUT_SIZE x INPUT_SIZE
         antes da rede. O `dataset.py` entrega 32x32 e é compartilhado por todos os
         IDs, por isso a ampliação fica dentro do modelo.
      2. BatchNorm congelado em modo eval: toda camada BatchNorm cujos pesos estão
         congelados (requires_grad=False) permanece em eval() mesmo depois de
         `model.train()`.

    Justificativa:
      - Resolução: a ResNet18 reduz a resolução espacial em 32x. Uma entrada de
        32x32 chega ao último bloco como um mapa 1x1, e os filtros, aprendidos
        com objetos em imagens de 224x224, nunca veem o objeto na escala em que
        foram treinados.
      - BatchNorm: `requires_grad=False` congela apenas gama e beta. A média e a
        variância correntes são buffers, atualizados a cada forward em modo de
        treino, e o `engine.py` chama `model.train()` a cada época. Sem este
        cuidado o backbone "congelado" passaria a normalizar pelas estatísticas
        do lote e teria as estatísticas da ImageNet sobrescritas.
      - O critério é por camada (pesos do BatchNorm congelados), não pelo backbone
        inteiro: no fine-tuning as camadas descongeladas seguem treinando
        normalmente, e um Dropout na cabeça não é afetado.

    Sondagem preliminar (regressão logística sobre as features da ResNet18
    congelada; 4.000 imagens de treino e 1.000 de validação, ambas tiradas do
    conjunto de treino; conjunto de teste intocado):
      - Acurácia de validação por resolução de entrada: 53,6% em 32px, 67,0% em
        64px, 76,3% em 96px, 79,1% em 128px e 82,2% em 224px.
      - Deixando o BatchNorm se adaptar em 32px, a acurácia caiu de 53,6% para 36,5%.
      - Custo do forward em CPU: 1.251 img/s em 32px, 140 em 96px e 32 em 224px.
      - Normalização: mantida a do CIFAR-10 que o `dataset.py` já aplica. Trocar
        pela da ImageNet não mudou o resultado em 96px (76,3% contra 76,4%).

    Efeito colateral (limitação do experimento):
      - A ampliação não cria informação: a imagem continua com o detalhe de 32x32.
        O ganho vem de casar a escala dos objetos com a do pré-treino.
      - O custo por imagem cresce com a área da entrada (cerca de 39x entre 32px
        e 224px na medição acima), tanto no treino quanto na inferência.
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
    O ID 4 já aplica a sua estratégia (backbone congelado). Os IDs 5 a 7 ainda
    carregam as redes com pesos da ImageNet sem congelar camadas (Fase A).
    """
    # --------------------------------------------------------------------------
    # REDE PRONTA 1: ResNet18 (IDs 4 e 5)
    # --------------------------------------------------------------------------
    if exp_id == 4:
        # Estratégia 1: backbone congelado, só a camada `fc` (512 -> 10) treina
        model = _build_resnet18(num_classes=num_classes, freeze_backbone=True)
        return model, "ID4_ResNet18_FeatExtract"

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