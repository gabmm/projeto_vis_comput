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
      'optimizer', 'loss_fn' e 'weight_decay'.
    - device (str): Dispositivo de execução ('cuda' ou 'cpu').

OUTPUTS (Arquivos gerados em disco e retorno em memória):
    - Arquivo de Pesos (.pth): Salvo em 'results/weights/exp_{exp_id}_best.pth'
      contendo o estado do modelo na época de maior acurácia de validação.
    - Arquivo de Log (.json): Salvo em 'results/logs/exp_{exp_id}.json' contendo:
        * exp_id, exp_name, hyperparameters, convergence_status
        * total_params, trainable_params
        * train_time_sec, inference_time_ms_per_image
        * best_val_acc, best_epoch, final_val_acc
        * history: listas por época de [train_loss, val_loss, train_acc, val_acc]
    - Retorno da função (dict): O próprio dicionário de métricas recém-salvo.
================================================================================
"""

import os
import json
import time
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """
    Treina o modelo por 1 época nas 45.000 imagens de treinamento.
    Retorna (train_loss_medio, train_acc_percentual).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += batch_size
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """
    Avalia o modelo nas 5.000 imagens de validação (sem atualização de gradiente).
    Retorna (val_loss_medio, val_acc_percentual, inference_time_ms_por_imagem).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_infer = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_infer_sec = time.perf_counter() - start_infer

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    # Tempo médio de inferência em milissegundos por imagem individual
    inference_time_ms = (total_infer_sec * 1000.0) / total

    return val_loss, val_acc, inference_time_ms


def check_convergence(history: Dict[str, list]) -> str:
    """
    Analisa as curvas de perda e acurácia de treino vs. validação para diagnosticar
    o comportamento de convergência do modelo (Overfitting, Underfitting, etc.).
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    if not train_loss or any(math.isnan(x) or math.isinf(x) for x in train_loss):
        return "Diverged (NaN/Inf detectado na função de perda)"

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    best_val_acc = max(val_acc)
    acc_gap = final_train_acc - final_val_acc

    # Caso 1: Rede não saiu do chute aleatório (~10% para 10 classes) ou aprendeu muito pouco
    if max(train_acc) < 35.0 and best_val_acc < 35.0:
        return (
            f"Underfitting (Acurácia baixa no treino [{final_train_acc:.1f}%] "
            f"e na validação [{final_val_acc:.1f}%])"
        )

    # Caso 2: Overfitting (Treino descolou muito da validação ou val_loss subiu no final)
    min_val_loss = min(val_loss)
    val_loss_rising = (len(val_loss) >= 3) and (val_loss[-1] > min_val_loss * 1.15)
    if acc_gap > 12.0 or (acc_gap > 7.0 and val_loss_rising):
        return (
            f"Overfitting detectado (Gap Treino-Validação de {acc_gap:.1f}% "
            f"| Val Loss subiu de {min_val_loss:.4f} para {val_loss[-1]:.4f})"
        )

    # Caso 3: Comportamento saudável
    return (
        f"Convergência estável (Gap Treino-Validação controlado em {acc_gap:.1f}% "
        f"| Melhor Val Acc: {best_val_acc:.2f}%)"
    )


def train_and_validate(
    exp_id: int,
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyperparams: Dict[str, Any],
    device: str = "cuda",
    results_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Executa o loop completo de treinamento e validação, salva os melhores pesos (.pth)
    e o arquivo de telemetria (.json), e exibe a acurácia no conjunto de validação (5k).
    """
    # Garante que as pastas de saída existam
    logs_dir = os.path.join(results_dir, "logs")
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    model = model.to(device)

    # Contabilização de parâmetros (Totais vs. Treináveis)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Configuração da Função de Perda e Otimizador (apenas para parâmetros treináveis)
    criterion = nn.CrossEntropyLoss()
    lr = hyperparams.get("lr", 1e-3)
    weight_decay = hyperparams.get("weight_decay", 0.0)
    opt_name = hyperparams.get("optimizer", "Adam").lower()

    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_param_list, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_param_list, lr=lr, weight_decay=weight_decay
        )

    epochs = hyperparams.get("epochs", 10)

    print("=" * 80)
    print(f"[INÍCIO] Experimento ID {exp_id}: {exp_name} | Dispositivo: {device.upper()}")
    print(f"         Parâmetros Totais: {total_params:,} | Treináveis: {trainable_params:,}")
    print(f"         Épocas: {epochs} | LR: {lr} | Batch Size: {hyperparams.get('batch_size', 64)} | Opt: {opt_name.upper()}")
    print("-" * 80)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    best_epoch = 0
    best_weights_path = os.path.join(weights_dir, f"exp_{exp_id}_best.pth")
    last_infer_ms = 0.0

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, infer_ms = evaluate_validation(
            model, val_loader, criterion, device
        )
        last_infer_ms = infer_ms

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))

        # Salva os pesos sempre que a acurácia nas 5k imagens de validação bater recorde
        saved_flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)
            saved_flag = " [* Melhor Val Acc]"

        epoch_elapsed = time.perf_counter() - epoch_t0
        print(
            f"Época [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% || "
            f"Val Loss (5k): {val_loss:.4f} | Val Acc (5k): {val_acc:6.2f}%{saved_flag}"
        )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_train_time_sec = time.perf_counter() - train_start_time

    convergence_status = check_convergence(history)

    # Monta o contrato padronizado do JSON para consumo futuro pelo evaluate.py
    metrics_data = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "device": device,
        "hyperparameters": hyperparams,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_sec": round(total_train_time_sec, 2),
        "inference_time_ms_per_image": round(last_infer_ms, 4),
        "best_val_acc": round(best_val_acc, 2),
        "best_epoch": best_epoch,
        "final_val_acc": round(history["val_acc"][-1], 2),
        "convergence_status": convergence_status,
        "weights_path": best_weights_path,
        "history": history
    }

    log_path = os.path.join(logs_dir, f"exp_{exp_id}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    print("-" * 80)
    print(f"[RESUMO ID {exp_id} - {exp_name}]")
    print(f"  -> Acurácia na Validação (5k imagens - Final):  {history['val_acc'][-1]:.2f}%")
    print(f"  -> Acurácia na Validação (5k imagens - Melhor): {best_val_acc:.2f}% (Época {best_epoch})")
    print(f"  -> Diagnóstico de Convergência: {convergence_status}")
    print(f"  -> Tempo Total de Treino: {total_train_time_sec:.2f}s | Inferência: {last_infer_ms:.4f} ms/img")
    print(f"  -> Arquivos salvos: '{log_path}' e '{best_weights_path}'")
    print("=" * 80 + "\n")

    return metrics_data


# ==============================================================================
# BLOCO DE AUTOTESTE (Permite testar o motor isoladamente em 2 segundos)
# Como rodar: python3 src/engine.py
# ==============================================================================
if __name__ == "__main__":
    print("[TESTE] Executando teste rápido do src/engine.py com tensores sintéticos...\n")
    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cria um dataset sintético minúsculo (32 imagens de treino, 16 de validação)
    dummy_train_ds = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
    dummy_val_ds = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))
    dummy_train_loader = DataLoader(dummy_train_ds, batch_size=16, shuffle=True)
    dummy_val_loader = DataLoader(dummy_val_ds, batch_size=16, shuffle=False)

    # Rede mínima apenas para validar o loop e o salvamento do JSON
    tiny_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dummy_hparams = {"epochs": 2, "lr": 1e-3, "batch_size": 16, "optimizer": "Adam"}

    result = train_and_validate(
        exp_id=999,
        exp_name="Teste_Unitario_Engine",
        model=tiny_model,
        train_loader=dummy_train_loader,
        val_loader=dummy_val_loader,
        hyperparams=dummy_hparams,
        device=test_device
    )

    # Remove os arquivos temporários do teste unitário (exp_999)
    for temp_file in ["./results/logs/exp_999.json", "./results/weights/exp_999_best.pth"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print("[OK] Autoteste do engine.py concluído e arquivos temporários limpos com sucesso!")