"""
================================================================================
ARQUIVO: evaluate.py
================================================================================
PAPEL NO PROJETO:
    Módulo de pós-processamento, avaliação final no conjunto de Teste e geração
    de artefatos visuais para o artigo em LaTeX. Opera em dois modos distintos
    para respeitar o isolamento do conjunto de teste exigido no enunciado:
      1. Modo Validação ('val'): Lê todos os JSONs em 'results/logs/', gera as
         curvas de aprendizado (Loss e Acurácia de treino vs. validação) e
         imprime o ranking de validação para seleção dos 3 melhores modelos.
      2. Modo Teste Final ('test'): Carrega os pesos (.pth) APENAS dos 3 modelos
         vencedores (1 CNN própria + 2 Transfer Learning), executa a inferência
         nas 10.000 imagens de teste e gera as Matrizes de Confusão e métricas
         por classe (Acurácia, Precision, Recall, F1-score).

INPUTS:
    - mode (str): 'val' (análise dos 8 experimentos) ou 'test' (avaliação final).
    - winner_ids (list[int], usado apenas quando mode='test'): Lista com os 3 IDs
      campeões escolhidos na validação (ex: [2, 5, 7]).
    - Arquivos lidos do disco: 'results/logs/exp_*.json',
      'results/weights/exp_*_best.pth' e o `test_loader` (10k imagens) vindo de
      `src/dataset.py`.

OUTPUTS (Salvos na pasta 'results/figures/'):
    - Gráficos vetoriais (.pdf / .eps): Curvas de perda/acurácia por época e
      Matrizes de Confusão das configurações finais.
    - Tabelas consolidadas (.csv / .tex / terminal): Comparativo de acurácia,
      número de parâmetros (totais vs. treináveis), tempo de treino e inferência.
================================================================================
"""