"""
================================================================================
ARQUIVO: evaluate.py
================================================================================
PAPEL NO PROJETO:
    Módulo de pós-processamento, avaliação final no conjunto de Teste e geração
    de artefatos visuais vetoriais (.pdf) para o artigo em LaTeX. Opera em dois
    modos distintos para respeitar o isolamento do conjunto de teste:
      1. Modo Validação ('val'): Lê todos os JSONs em 'results/logs/', plota as
         curvas de aprendizado (Loss e Acurácia de treino vs. validação) em PDF
         e imprime o ranking comparativo nas 5.000 imagens de validação.
      2. Modo Teste Final ('test'): Carrega os pesos (.pth) dos modelos campeões
         escolhidos na validação, executa a inferência nas 10.000 imagens de
         teste e gera as Matrizes de Confusão em PDF e métricas por classe.

COMO EXECUTAR NO TERMINAL:
    - Analisar os modelos já treinados na Validação (5k) e gerar curvas PDF:
        python3 evaluate.py --mode val
    - Avaliar os 3 campeões no Conjunto de Teste (10k) e gerar Matrizes PDF:
        python3 evaluate.py --mode test --winners 0 4 6

INPUTS:
    - --mode (str): 'val' ou 'test'.
    - --winners (list[int], obrigatório se mode='test'): IDs vencedores.

OUTPUTS (Salvos em './results/figures/'):
    - 'learning_curves_exp_{id}.pdf': Curvas de Loss e Acurácia por experimento.
    - 'confusion_matrix_exp_{id}.pdf': Matriz de confusão no conjunto de teste.
    - 'final_test_comparison.json': Consolidação das métricas finais de teste.
================================================================================
"""

import os
import glob
import json
import time
import argparse
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import get_dataloaders
from src.models_custom import get_custom_model
from src.models_transfer import get_transfer_model

RESULTS_DIR = "./results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def plot_learning_curves(log_data: Dict[str, Any], output_dir: str) -> str:
    """
    Gera e salva um arquivo vetorial (.pdf) contendo lado a lado as curvas de
    Perda (Loss) e Acurácia de Treinamento vs. Validação ao longo das épocas.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_id = log_data["exp_id"]
    exp_name = log_data["exp_name"]
    history = log_data["history"]
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Comportamento da Função de Perda (Loss)
    ax1.plot(epochs, history["train_loss"], label="Treino (45k)", marker="o", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validação (5k)", marker="s", linewidth=2, linestyle="--")
    ax1.set_title(f"{exp_name} - Função de Perda (Loss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend()

    # Subplot 2: Comportamento da Acurácia
    ax2.plot(epochs, history["train_acc"], label="Treino (45k)", marker="o", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Validação (5k)", marker="s", linewidth=2, linestyle="--")
    ax2.set_title(f"{exp_name} - Acurácia (%)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia (%)")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f"learning_curves_exp_{exp_id}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    exp_id: int,
    exp_name: str,
    test_acc: float,
    output_dir: str
) -> str:
    """
    Plota e salva a Matriz de Confusão em formato vetorial (.pdf) para as 10
    classes do CIFAR-10 avaliadas no conjunto de teste (10.000 imagens).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Matriz de Confusão (Teste 10k) - {exp_name} (Acc: {test_acc:.2f}%)",
        ylabel="Classe Verdadeira (Ground Truth)",
        xlabel="Classe Predita pelo Modelo"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anota os valores numéricos dentro de cada célula da matriz
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    pdf_path = os.path.join(output_dir, f"confusion_matrix_exp_{exp_id}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def summarize_validation_results(logs_dir: str = LOGS_DIR) -> None:
    """
    Lê todos os arquivos exp_*.json na pasta de logs, gera os gráficos PDF de
    curvas de aprendizado e imprime a tabela comparativa nas 5k imagens de validação.
    """
    json_files = sorted(glob.glob(os.path.join(logs_dir, "exp_*.json")))
    if not json_files:
        print(f"[AVISO] Nenhum arquivo JSON encontrado em '{logs_dir}'. Rode o run_project.py primeiro.")
        return

    logs_data = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            logs_data.append(data)

    logs_data.sort(key=lambda x: x["exp_id"])

    print("\n" + "=" * 115)
    print("RANKING DE VALIDAÇÃO (5.000 IMAGENS) E TELEMETRIA COMPUTACIONAL")
    print("=" * 115)
    header = (
        f"{'ID':<3} | {'Experimento':<28} | {'Melhor Val (5k)':<15} | "
        f"{'Params Totais':<14} | {'Params Trein.':<14} | {'Treino (s)':<10} | {'Infer (ms/img)':<14}"
    )
    print(header)
    print("-" * 115)

    for log in logs_data:
        pdf_curve = plot_learning_curves(log, FIGURES_DIR)
        print(
            f"{log['exp_id']:<3} | {log['exp_name']:<28} | "
            f"{log['best_val_acc']:>6.2f}% (Ep {log['best_epoch']:02d}) | "
            f"{log['total_params']:>14,} | {log['trainable_params']:>14,} | "
            f"{log['train_time_sec']:>10.2f} | {log['inference_time_ms_per_image']:>14.4f}"
        )

    print("=" * 115)
    print("\n[DIAGNÓSTICO DE CONVERGÊNCIA POR EXPERIMENTO]")
    for log in logs_data:
        print(f"  * ID {log['exp_id']} ({log['exp_name']}): {log['convergence_status']}")

    # Sugestão automática dos vencedores por categoria com base na validação
    part1 = [x for x in logs_data if 0 <= x["exp_id"] <= 3]
    trans_a = [x for x in logs_data if x["exp_id"] in (4, 5)]
    trans_b = [x for x in logs_data if x["exp_id"] in (6, 7)]

    print("\n[DESTAQUES ATUAIS NA VALIDAÇÃO (Para seleção rumo ao Teste Final)]")
    if part1:
        best_p1 = max(part1, key=lambda x: x["best_val_acc"])
        print(f"  -> Melhor CNN Própria (IDs 0-3):        ID {best_p1['exp_id']} ({best_p1['exp_name']}) com {best_p1['best_val_acc']:.2f}%")
    if trans_a:
        best_ta = max(trans_a, key=lambda x: x["best_val_acc"])
        print(f"  -> Melhor Transfer Rede 1 (IDs 4-5):    ID {best_ta['exp_id']} ({best_ta['exp_name']}) com {best_ta['best_val_acc']:.2f}%")
    if trans_b:
        best_tb = max(trans_b, key=lambda x: x["best_val_acc"])
        print(f"  -> Melhor Transfer Rede 2 (IDs 6-7):    ID {best_tb['exp_id']} ({best_tb['exp_name']}) com {best_tb['best_val_acc']:.2f}%")

    print(f"\n[OK] Curvas de aprendizado em formato vetorial (.pdf) salvas em: '{FIGURES_DIR}/'\n")


def run_final_test(winner_ids: List[int]) -> None:
    """
    Carrega os pesos (.pth) dos modelos vencedores, executa a inferência no
    conjunto de Teste isolado (10.000 imagens), plota as Matrizes de Confusão
    em PDF e gera a tabela comparativa final do artigo.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "#" * 95)
    print(f"AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (10.000 IMAGENS ISOLADAS) | Dispositivo: {device.upper()}")
    print(f"IDs Vencedores Selecionados: {winner_ids}")
    print("#" * 95 + "\n")

    # Carrega o test_loader (10k imagens) e os nomes das 10 classes
    _, _, test_loader, classes = get_dataloaders(batch_size=64, seed=42)
    criterion = nn.CrossEntropyLoss()

    final_test_summary = []

    for exp_id in winner_ids:
        log_path = os.path.join(LOGS_DIR, f"exp_{exp_id}.json")
        if not os.path.exists(log_path):
            raise FileNotFoundError(
                f"[ERRO] Log '{log_path}' não encontrado. Treine o ID {exp_id} antes de avaliá-lo no teste."
            )

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        weights_path = log_data["weights_path"]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[ERRO] Arquivo de pesos '{weights_path}' não encontrado.")

        # Reconstrói a arquitetura e carrega os melhores pesos obtidos na validação
        if 0 <= exp_id <= 3:
            model, exp_name = get_custom_model(exp_id=exp_id, num_classes=10)
        elif 4 <= exp_id <= 7:
            model, exp_name = get_transfer_model(exp_id=exp_id, num_classes=10)
        else:
            raise ValueError(f"[ERRO] exp_id={exp_id} fora do intervalo 0-7.")

        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        all_preds = []
        all_targets = []
        running_loss = 0.0
        total = 0

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total += batch_size

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_sec = time.perf_counter() - t0

        test_loss = running_loss / total
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)
        test_acc = 100.0 * np.mean(all_preds_np == all_targets_np)
        infer_ms_per_img = (infer_sec * 1000.0) / total

        # Gera a Matriz de Confusão e salva em PDF
        cm = confusion_matrix(all_targets_np, all_preds_np)
        cm_pdf_path = plot_confusion_matrix(
            cm=cm,
            classes=classes,
            exp_id=exp_id,
            exp_name=exp_name,
            test_acc=test_acc,
            output_dir=FIGURES_DIR
        )

        # Identifica o par de classes mais confundido (fora da diagonal principal)
        cm_off_diag = cm.copy()
        np.fill_diagonal(cm_off_diag, 0)
        max_conf_idx = np.unravel_index(np.argmax(cm_off_diag), cm_off_diag.shape)
        true_cls = classes[max_conf_idx[0]]
        pred_cls = classes[max_conf_idx[1]]
        conf_count = int(cm_off_diag[max_conf_idx])

        print("=" * 95)
        print(f"[RESULTADO NO TESTE (10k) - ID {exp_id}: {exp_name}]")
        print(f"  -> Acurácia no Teste (10k): {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
        print(f"  -> Maior Confusão entre Classes: '{true_cls}' classificado como '{pred_cls}' ({conf_count} vezes)")
        print(f"  -> Matriz de Confusão salva em: '{cm_pdf_path}'")
        print("\nRelatório por Classe (Precision / Recall / F1-Score):")
        print(classification_report(all_targets_np, all_preds_np, target_names=classes, digits=4))

        final_test_summary.append({
            "exp_id": exp_id,
            "exp_name": exp_name,
            "val_best_acc": log_data["best_val_acc"],
            "test_acc": round(test_acc, 2),
            "test_loss": round(test_loss, 4),
            "total_params": log_data["total_params"],
            "trainable_params": log_data["trainable_params"],
            "train_time_sec": log_data["train_time_sec"],
            "test_infer_ms_per_img": round(infer_ms_per_img, 4),
            "most_confused_pair": f"{true_cls} -> {pred_cls} ({conf_count})"
        })

    # Tabela Consolidada Final (Seção 3 do Enunciado: Análise Comparativa)
    print("\n" + "=" * 115)
    print("TABELA COMPARATIVA GLOBAL FINAL (CONJUNTO DE TESTE - 10.000 IMAGENS)")
    print("=" * 115)
    print(
        f"{'ID':<3} | {'Modelo':<28} | {'Val Acc (5k)':<12} | {'Test Acc (10k)':<14} | "
        f"{'Params Totais':<14} | {'Treino (s)':<10} | {'Maior Confusão':<22}"
    )
    print("-" * 115)
    for item in final_test_summary:
        print(
            f"{item['exp_id']:<3} | {item['exp_name']:<28} | "
            f"{item['val_best_acc']:>10.2f}% | {item['test_acc']:>12.2f}% | "
            f"{item['total_params']:>14,} | {item['train_time_sec']:>10.2f} | "
            f"{item['most_confused_pair']:<22}"
        )
    print("=" * 115 + "\n")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary_json_path = os.path.join(FIGURES_DIR, "final_test_comparison.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_test_summary, f, indent=4, ensure_ascii=False)
    print(f"[OK] Resumo comparativo final salvo em: '{summary_json_path}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avaliação de Validação, Teste Final e Geração de Gráficos Vetoriais (.pdf)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Modo de operação: 'val' (curvas e tabela nos 5k) ou 'test' (avaliação final nos 10k)"
    )
    parser.add_argument(
        "--winners",
        type=int,
        nargs="+",
        default=None,
        help="Lista de IDs vencedores para avaliar no conjunto de teste (ex: --winners 0 4 6)"
    )

    args = parser.parse_args()

    if args.mode == "val":
        summarize_validation_results()
    elif args.mode == "test":
        if not args.winners:
            parser.error("O modo '--mode test' exige a passagem dos IDs campeões via '--winners' (ex: --winners 0 4 6).")
        run_final_test(winner_ids=args.winners)