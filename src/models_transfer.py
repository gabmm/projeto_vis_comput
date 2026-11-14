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
    - ID 4: Arquitetura Pré-treinada A - Estratégia 1 (ex: Backbone congelado).
    - ID 5: Arquitetura Pré-treinada A - Estratégia 2 (ex: Fine-tuning parcial/total).
    - ID 6: Arquitetura Pré-treinada B - Estratégia 1 (ex: Backbone congelado).
    - ID 7: Arquitetura Pré-treinada B - Estratégia 2 (ex: Fine-tuning parcial/total).
    (Opcional: IDs 8 e 9 para treino do zero das mesmas arquiteturas sem pesos).

INPUTS (Parâmetros da função fábrica `get_transfer_model`):
    - exp_id (int): Identificador do experimento (valores aceitos: 4, 5, 6 ou 7+).
    - num_classes (int): Número de classes de saída (padrão: 10).

OUTPUTS (Retorno da função fábrica):
    - model (torch.nn.Module): Modelo pré-treinado adaptado e com gradientes
      travados/destravados conforme a estratégia do ID solicitado.
    - exp_name (str): Nome descritivo do experimento (ex: 'ID4_ResNet18_Frozen').
================================================================================
"""