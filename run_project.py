"""
================================================================================
ARQUIVO: run_project.py
================================================================================
PAPEL NO PROJETO:
    Ponto de entrada (Orquestrador) para execução dos treinamentos. Centraliza
    no cabeçalho os hiperparâmetros globais fixos (garantindo comparação justa
    entre os modelos) e recebe via linha de comando a lista de IDs de
    experimentos (0 a 7) para rodar em sequência, conectando `src/dataset.py`,
    os módulos de modelos (`models_custom.py` / `models_transfer.py`) e o motor
    de treinamento (`src/engine.py`).

COMO EXECUTAR NO TERMINAL:
    - Rodar apenas a Baseline (ID 0):
        python3 run_project.py --id 0
    - Rodar múltiplos IDs em sequência:
        python3 run_project.py --id 0 4 6
    - Teste rápido da Fase A (sobrescrevendo para 1 época apenas na execução):
        python3 run_project.py --id 0 1 4 6 --epochs 1

INPUTS (Argumentos CLI e Constantes de Cabeçalho):
    - --id (list[int]): Um ou mais IDs de experimentos (valores entre 0 e 7).
    - --epochs (int, opcional): Sobrescreve o número de épocas padrão para testes.
    - BASE_HYPERPARAMS (dict): Hiperparâmetros fixos compartilhados por todos.

OUTPUTS:
    - Salva automaticamente os pesos (.pth) e métricas (.json) em './results/'
      por intermédio do `src/engine.py`.
    - Imprime no terminal a acurácia obtida nas 5.000 imagens de validação.
================================================================================
"""

import argparse
import copy
import random
from typing import List, Dict, Any
import numpy as np
import torch

from src.dataset import get_dataloaders
from src.models_custom import get_custom_model
from src.models_transfer import get_transfer_model
from src.engine import train_and_validate


# ==============================================================================
# CABEÇALHO DE CONFIGURAÇÕES E HIPERPARÂMETROS FIXOS DO PROJETO
# (Mantidos constantes entre os experimentos para garantir comparação justa)
# ==============================================================================
SEED: int = 42
DATA_DIR: str = "./data"
RESULTS_DIR: str = "./results"

BASE_HYPERPARAMS: Dict[str, Any] = {
    "batch_size": 64,
    "epochs": 20,
    "lr": 1e-3,
    "optimizer": "Adam",
    "loss_fn": "CrossEntropyLoss",
    "weight_decay": 0.0,
    "seed": SEED
}

# Caso algum experimento da Fase B exija ajuste pontual de hiperparâmetro
# (ex: reduzir LR no fine-tuning completo do ID 5 ou 7), documente abaixo:
EXP_OVERRIDES: Dict[int, Dict[str, Any]] = {
    # Exemplo para Fase B:
    # 5: {"lr": 1e-4},
    # 7: {"lr": 1e-4},
}
# ==============================================================================


def set_global_seed(seed: int) -> None:
    """Fixa todas as sementes aleatórias para garantir reprodutibilidade bit a bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiments(exp_ids: List[int], epochs_override: int = None) -> None:
    """
    Orquestra o carregamento dos dados, instanciação dos modelos e treinamento
    para cada ID fornecido na lista `exp_ids`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "#" * 80)
    print(f"ORQUESTRADOR DE EXPERIMENTOS CIFAR-10 | Dispositivo: {device.upper()}")
    print(f"IDs agendados para esta execução: {exp_ids}")
    print("#" * 80 + "\n")

    # 1. Carrega os DataLoaders uma única vez com a divisão estratificada fixa (45k / 5k)
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=BASE_HYPERPARAMS["batch_size"],
        seed=SEED,
        data_dir=DATA_DIR
    )

    summary_results = []

    # 2. Percorre cada ID solicitado
    for exp_id in exp_ids:
        set_global_seed(SEED)

        # Copia os hiperparâmetros base e aplica eventuais overrides
        current_hparams = copy.deepcopy(BASE_HYPERPARAMS)
        if exp_id in EXP_OVERRIDES:
            current_hparams.update(EXP_OVERRIDES[exp_id])
        if epochs_override is not None:
            current_hparams["epochs"] = epochs_override

        # Roteamento: Parte 1 (IDs 0 a 3) vs. Parte 2 (IDs 4 a 7)
        if 0 <= exp_id <= 3:
            model, exp_name = get_custom_model(exp_id=exp_id, num_classes=10)
        elif 4 <= exp_id <= 7:
            model, exp_name = get_transfer_model(exp_id=exp_id, num_classes=10)
        else:
            raise ValueError(f"[ERRO] ID {exp_id} inválido! Os IDs válidos vão de 0 a 7.")

        # 3. Executa o treinamento e a avaliação nas 5.000 imagens de validação
        metrics = train_and_validate(
            exp_id=exp_id,
            exp_name=exp_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hyperparams=current_hparams,
            device=device,
            results_dir=RESULTS_DIR
        )

        summary_results.append(metrics)

    # 4. Exibe um quadro consolidado ao final de todos os IDs rodados
    print("\n" + "=" * 80)
    print("RESUMO FINAL DA BATERIA DE EXECUÇÃO (Avaliado nas 5.000 imagens de Validação)")
    print("=" * 80)
    print(f"{'ID':<4} | {'Experimento':<30} | {'Melhor Val Acc (5k)':<20} | {'Final Val Acc (5k)':<18}")
    print("-" * 80)
    for res in summary_results:
        print(
            f"{res['exp_id']:<4} | {res['exp_name']:<30} | "
            f"{res['best_val_acc']:>6.2f}% (Época {res['best_epoch']:02d})   | "
            f"{res['final_val_acc']:>6.2f}%"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orquestrador de Treinamento e Validação - Trabalho Prático CIFAR-10"
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        required=True,
        help="ID(s) do(s) experimento(s) para rodar (ex: --id 0 ou --id 0 1 4 6)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve temporariamente o número de épocas (útil para testes rápidos)"
    )

    args = parser.parse_args()
    run_experiments(exp_ids=args.id, epochs_override=args.epochs)