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
    - ID 1: Modificação Arquitetural 1 (ex: inclusão de Batch Normalization).
    - ID 2: Modificação Arquitetural 2 (ex: aumento de profundidade / filtros).
    - ID 3: Modificação Arquitetural 3 (ex: inclusão de Dropout / ajuste denso).

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