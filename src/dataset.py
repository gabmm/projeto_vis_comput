"""
================================================================================
ARQUIVO: src/dataset.py
================================================================================
PAPEL NO PROJETO:
    Responsável pelo download, pré-processamento, Data Augmentation (se houver)
    e divisão estratificada fixa do dataset CIFAR-10. Garante que todos os
    experimentos (Parte 1 e Parte 2) utilizem exatamente a mesma partição de
    dados, mantendo o conjunto de teste isolado até a avaliação final.

INPUTS (Parâmetros da função principal `get_dataloaders`):
    - batch_size (int): Tamanho do lote de imagens por iteração.
    - seed (int): Semente aleatória fixa (ex: 42) para garantir reprodutibilidade
      na divisão estratificada em todos os computadores do grupo.
    - data_dir (str): Caminho da pasta onde os binários do CIFAR-10 serão salvos
      (padrão: './data').
    - num_workers (int): Número de subprocessos para carregamento de dados.

OUTPUTS (Retorno da função principal):
    - train_loader (DataLoader): 45.000 imagens de treinamento (estratificado).
    - val_loader (DataLoader): 5.000 imagens de validação (estratificado).
    - test_loader (DataLoader): 10.000 imagens de teste oficiais do CIFAR-10
      (NÃO deve ser utilizado durante o treinamento ou escolha de modelos).
    - classes (list[str]): Lista com os nomes das 10 classes do CIFAR-10.
================================================================================
"""

from typing import Tuple, List
from collections import Counter
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Estatísticas clássicas (média e desvio padrão por canal RGB) do CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    batch_size: int = 64,
    seed: int = 42,
    data_dir: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Baixa o CIFAR-10, aplica normalização e faz o split estratificado fixo:
    - train_loader: 45.000 imagens
    - val_loader: 5.000 imagens (usado para retornar a acurácia ao fim de cada ID)
    - test_loader: 10.000 imagens (isolado apenas para o evaluate.py no final)
    - classes: lista com os 10 nomes das classes
    """
    # 1. Definição das transformações de pré-processamento
    # Nota: Data Augmentation (ex: RandomHorizontalFlip) pode ser
    # adicionado em `train_transform` sem afetar `eval_transform`.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # 2. Download e carregamento do conjunto original de treino (50.000) e teste (10.000)
    # Criamos duas instâncias do dataset de treino para isolar as transformações de treino e validação
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_val_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # 3. Divisão estratificada fixa: 45.000 para treino (90%) e 5.000 para validação (10%)
    targets = full_train_dataset.targets  # Lista com os 50.000 rótulos originais (0 a 9)
    all_indices = list(range(len(targets)))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=5000,
         random_state=seed,
        stratify=targets
    )

    # 4. Criação dos Subsets aplicando os índices estratificados
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_val_dataset, val_idx)

    # 5. Criação dos DataLoaders
    # Fixamos o generator do shuffle de treino com a seed para reprodutibilidade
    g = torch.Generator()
    g.manual_seed(seed)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    classes = full_train_dataset.classes

    return train_loader, val_loader, test_loader, classes


# ==============================================================================
# BLOCO DE AUTOTESTE (Permite testar o arquivo isoladamente no terminal)
# Como rodar: python src/dataset.py
# ==============================================================================
if __name__ == "__main__":
    print("[TESTE] Inicializando download e divisão estratificada do CIFAR-10...")
    train_ldr, val_ldr, test_ldr, class_names = get_dataloaders(batch_size=64, seed=42)

    print(f"\n[OK] Classes ({len(class_names)}): {class_names}")
    print(f"[OK] Total de imagens de Treino:    {len(train_ldr.dataset)} (Esperado: 45000)")
    print(f"[OK] Total de imagens de Validação: {len(val_ldr.dataset)} (Esperado: 5000)")
    print(f"[OK] Total de imagens de Teste:     {len(test_ldr.dataset)} (Esperado: 10000)")

    # Verifica se a estratificação nas 5.000 imagens de validação ficou exata (500 por classe)
    val_targets = [val_ldr.dataset.dataset.targets[i] for i in val_ldr.dataset.indices]
    dist_val = Counter(val_targets)
    print(f"[OK] Distribuição por classe na Validação (5k): {dict(sorted(dist_val.items()))}")

    # Verifica o formato de 1 batch
    images, labels = next(iter(train_ldr))
    print(f"[OK] Shape de 1 batch de imagens: {images.shape} | Labels: {labels.shape}")