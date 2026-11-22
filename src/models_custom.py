"""
================================================================================
ARQUIVO: src/models_custom.py
================================================================================
PAPEL NO PROJETO:
    Contém as definições arquiteturais da Etapa 1.1 (Desenvolvimento da CNN
    própria). Implementa o Modelo Baseline e os 3 experimentos controlados de
    modificação arquitetural (alterando uma característica por vez).

MAPEAMENTO DE IDs DESTE MÓDULO:
    - ID 0: CNN Baseline (arquitetura de referência).
    - ID 1: Modificação Arquitetural 1 (Baseline + Batch Normalization após cada convolução).
    - ID 2: Modificação Arquitetural 2 (ID 1 + Dropout de 40% no classificador).
    - ID 3: Modificação Arquitetural 3 (ID 2 + terceiro bloco convolucional com 128 filtros).

INPUTS (Parâmetros da função fábrica `get_custom_model`):
    - exp_id (int): Identificador do experimento (valores aceitos: 0, 1, 2 ou 3).
    - num_classes (int): Número de classes de saída (padrão: 10 para CIFAR-10).

OUTPUTS (Retorno da função fábrica):
    - model (torch.nn.Module): Instância da rede neural convolucional em PyTorch
      pronta para ser enviada à GPU/CPU.
    - exp_name (str): Nome descritivo do experimento (ex: 'ID0_Baseline_CNN')
      para registro nos logs e gráficos.
================================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Arquitetura Baseline (ID 0) para classificação do CIFAR-10 (3x32x32).
    Estrutura:
      - Bloco Conv 1: Conv2d(3 -> 32, k=3, p=1) + ReLU + MaxPool2d(2x2) -> Saída: 32x16x16
      - Bloco Conv 2: Conv2d(32 -> 64, k=3, p=1) + ReLU + MaxPool2d(2x2) -> Saída: 64x8x8
      - Classificador: Flatten(4096) -> Linear(4096 -> 128) + ReLU -> Linear(128 -> 10)
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco Convolucional 1 (Entrada: 3 x 32 x 32 -> Saída: 32 x 16 x 16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco Convolucional 2 (Entrada: 32 x 16 x 16 -> Saída: 64 x 8 x 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class Exp1BatchNormCNN(nn.Module):
    """
    Arquitetura Baseline (ID 1) para classificação do CIFAR-10 (3x32x32).
    Estrutura:
        TODO
        Exemplo: Incluir Batch Normalization após as convoluções
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco Convolucional 1 (Entrada: 3 x 32 x 32 -> Saída: 32 x 16 x 16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco Convolucional 2 (Entrada: 32 x 16 x 16 -> Saída: 64 x 8 x 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class Exp2BatchNormCNN(nn.Module):
    """
    Arquitetura Baseline (ID 2) para classificação do CIFAR-10 (3x32x32).
    Estrutura:
        TODO 
        Exemplo: Aumentar a profundidade (3º bloco conv com 128 filtros)
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco Convolucional 1 (Entrada: 3 x 32 x 32 -> Saída: 32 x 16 x 16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco Convolucional 2 (Entrada: 32 x 16 x 16 -> Saída: 64 x 8 x 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),  # Adicionando Dropout
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class Exp3BatchNormCNN(nn.Module):
    """
    Arquitetura do ID 3 para classificação do CIFAR-10 (3x32x32):
    ID 2 (BatchNorm + Dropout) + 3º bloco convolucional com 128 filtros.
    Estrutura:
      - Bloco Conv 1: Conv2d(3 -> 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) -> Saída: 32x16x16
      - Bloco Conv 2: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) -> Saída: 64x8x8
      - Bloco Conv 3: Conv2d(64 -> 128, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2x2) -> Saída: 128x4x4
      - Classificador: Flatten(2048) -> Linear(2048 -> 128) + ReLU + Dropout(p=0.4) -> Linear(128 -> 10)

    O bloco novo aumenta o campo receptivo de 10x10 para 22x22 pixels. O MaxPool
    extra reduz a entrada do primeiro Linear de 4096 para 2048, então o total de
    parâmetros cai de 545.290 para 357.258: profundidade e tamanho da camada densa
    mudam juntos.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco Convolucional 1 (Entrada: 3 x 32 x 32 -> Saída: 32 x 16 x 16)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco Convolucional 2 (Entrada: 32 x 16 x 16 -> Saída: 64 x 8 x 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco Convolucional 3 (Entrada: 64 x 8 x 8 -> Saída: 128 x 4 x 4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Entrada cai de 4096 (ID 2) para 2048 por causa do MaxPool do bloco 3
            nn.Linear(in_features=128 * 4 * 4, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_custom_model(exp_id: int, num_classes: int = 10) -> Tuple[nn.Module, str]:
    """
    Fábrica de modelos (IDs 0 a 3).
    """
    if exp_id == 0:
        return BaselineCNN(num_classes=num_classes), "ID0_Baseline_CNN"

    elif exp_id == 1:
        return Exp1BatchNormCNN(num_classes=num_classes), "ID1_Custom_Mod1_Stub"

    elif exp_id == 2:
        return Exp2BatchNormCNN(num_classes=num_classes), "ID2_Custom_Mod2_Stub"

    elif exp_id == 3:
        return Exp3BatchNormCNN(num_classes=num_classes), "ID3_BN_Dropout_3Blocos"

    else:
        raise ValueError(
            f"[ERRO] exp_id={exp_id} inválido para models_custom.py. Use valores entre 0 e 3."
        )


# ==============================================================================
# BLOCO DE AUTOTESTE (Permite testar o arquivo isoladamente no terminal)
# Como rodar: python3 src/models_custom.py
# ==============================================================================
if __name__ == "__main__":
    print("[TESTE] Validando instâncias de src/models_custom.py (IDs 0 a 3)...\n")
    dummy_input = torch.randn(4, 3, 32, 32)  # Lote falso de 4 imagens RGB 32x32

    for current_id in range(4):
        model, name = get_custom_model(exp_id=current_id, num_classes=10)
        output = model(dummy_input)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"-> Experimento: {name} (ID {current_id})")
        print(f"   Shape de Saída: {output.shape} (Esperado: torch.Size([4, 10]))")
        print(f"   Parâmetros Totais: {total_params:,} | Treináveis: {trainable_params:,}\n")