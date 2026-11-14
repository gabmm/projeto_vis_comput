"""
================================================================================
ARQUIVO: run_project.py
================================================================================
PAPEL NO PROJETO:
    Ponto de entrada (Orquestrador) para execução dos treinamentos. Define os
    hiperparâmetros globais fixos (garantindo comparação justa entre os modelos)
    e recebe uma lista de IDs de experimentos para rodar em sequência,
    conectando o `dataset.py`, os módulos de `models` e o `engine.py`.

INPUTS (Configuração / Argumentos de execução):
    - exp_ids (list[int]): Lista de IDs que devem ser treinados nesta execução
      (ex: [0, 3, 4] ou [0, 1, 2, 3, 4, 5, 6, 7]).
    - HYPERPARAMS (dict constante no script): Definições fixas de learning rate,
      batch size, número de épocas, função de perda e otimizador.

OUTPUTS:
    - Não retorna variáveis (script de nível superior).
    - Dispara a criação dos arquivos '.json' e '.pth' dentro da pasta 'results/'
      por intermédio do `src/engine.py` e exibe logs de progresso no terminal.
================================================================================
"""