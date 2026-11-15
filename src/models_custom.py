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
    - ID 1: Modificação Arquitetural 1 (Fase A: clone da Baseline | Fase B: a definir).
    - ID 2: Modificação Arquitetural 2 (Fase A: clone da Baseline | Fase B: a definir).
    - ID 3: Modificação Arquitetural 3 (Fase A: clone da Baseline | Fase B: a definir).

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

class Exp3BatchNormCNN(nn.Module):
    """
    Arquitetura Baseline (ID 3) para classificação do CIFAR-10 (3x32x32).
    Estrutura:
        TODO
        Exemplo: Adicionar Dropout nas camadas densas ou conexões residuais
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
        return Exp3BatchNormCNN(num_classes=num_classes), "ID3_Custom_Mod3_Stub"

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