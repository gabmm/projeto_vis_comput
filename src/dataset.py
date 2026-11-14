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
    - img_size (int, opcional): Resolução espacial de saída (32x32 nativo para a
      CNN própria ou redimensionado caso alguma arquitetura da ImageNet exija).

OUTPUTS (Retorno da função principal):
    - train_loader (DataLoader): 45.000 imagens de treinamento (estratificado).
    - val_loader (DataLoader): 5.000 imagens de validação (estratificado).
    - test_loader (DataLoader): 10.000 imagens de teste oficiais do CIFAR-10
      (NÃO deve ser utilizado durante o treinamento ou escolha de modelos).
    - classes (list[str]): Lista com os nomes das 10 classes do CIFAR-10.
================================================================================
"""