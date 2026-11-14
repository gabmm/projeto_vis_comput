"""
================================================================================
ARQUIVO: src/engine.py
================================================================================
PAPEL NO PROJETO:
    Motor unificado de treinamento, validação e telemetria. É agnóstico à
    arquitetura (recebe qualquer modelo da Parte 1 ou Parte 2). Executa o loop
    de épocas, monitora overfitting/underfitting, cronometra tempo de treino e
    inferência, contabiliza parâmetros e salva os resultados padronizados em disco.

INPUTS (Parâmetros da função principal `train_and_validate`):
    - exp_id (int): ID numérico do experimento (0 a 7).
    - exp_name (str): Nome descritivo do experimento.
    - model (torch.nn.Module): Rede neural já instanciada.
    - train_loader (DataLoader): Dados de treinamento (45.000 imagens).
    - val_loader (DataLoader): Dados de validação (5.000 imagens).
    - hyperparams (dict): Dicionário contendo 'epochs', 'lr', 'batch_size',
      'optimizer', 'loss_fn' e 'lr_scheduler'.
    - device (str): Dispositivo de execução ('cuda' ou 'cpu').

OUTPUTS (Arquivos gerados em disco e retorno em memória):
    - Arquivo de Pesos (.pth): Salvo em 'results/weights/exp_{exp_id}_best.pth'
      contendo o estado do modelo na época de maior acurácia de validação.
    - Arquivo de Log (.json): Salvo em 'results/logs/exp_{exp_id}.json' contendo:
        * exp_id, exp_name, hyperparameters
        * total_params, trainable_params
        * train_time_sec, inference_time_ms
        * best_val_acc, best_epoch
        * history: listas por época de [train_loss, val_loss, train_acc, val_acc]
    - Retorno da função (dict): O próprio dicionário de métricas recém-salvo.
================================================================================
"""