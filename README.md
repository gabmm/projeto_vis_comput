# Classificação de Imagens no CIFAR-10: CNN Autoral vs. Transfer Learning

**Universidade Federal de Juiz de Fora (UFJF)**  
**Instituto de Ciências Exatas — Departamento de Ciência da Computação**  
**Disciplina:** Visão Computacional  
**Professor:** Luiz Maurílio da Silva Maciel  

---

## 1. Introdução

Este repositório contém a implementação experimental de Redes Neurais Convolucionais (CNNs) aplicadas ao problema de classificação de imagens no dataset **CIFAR-10** ($50.000$ imagens de treino e $10.000$ imagens de teste distribuídas em $10$ classes).

O estudo é dividido em duas frentes principais:
1. **Desenvolvimento de CNN Própria (IDs `0` a `3`):** Concepção de uma arquitetura *baseline* autoral e execução de três experimentos controlados de modificação arquitetural (alterando uma característica por vez e mantendo os hiperparâmetros constantes).
2. **Transfer Learning (IDs `4` a `7`):** Adaptação de duas arquiteturas consagradas da literatura pré-treinadas no dataset *ImageNet*, avaliando diferentes estratégias de treinamento (congelamento do *backbone* convolucional vs. descongelamento parcial/*fine-tuning*).

Para garantir rigor experimental e reprodutibilidade, o conjunto original de treinamento é dividido de forma estratificada e fixa em **$45.000$ imagens de treino** e **$5.000$ imagens de validação**. O conjunto de **$10.000$ imagens de teste** permanece estritamente isolado durante todo o desenvolvimento, sendo utilizado uma única vez ao final para avaliar os 3 modelos campeões (melhor CNN própria + melhores configurações de cada arquitetura pré-treinada).

---

## 2. Esqueleto do Projeto (Estrutura de Diretórios)

```text
projeto_vis_comput/
│
├── data/                        # Criada automaticamente (guarda os binários do CIFAR-10 baixados)
│
├── results/                     # Saídas padronizadas geradas pelo engine.py e evaluate.py
│   ├── logs/                    # Arquivos exp_0.json até exp_7.json (histórico, tempos, parâmetros)
│   ├── weights/                 # Arquivos exp_0_best.pth até exp_7_best.pth (pesos da melhor época)
│   └── figures/                 # Gráficos vetoriais (.pdf / .eps) e tabelas prontas para o LaTeX
│
├── src/                         # Código-fonte modularizado
│   ├── dataset.py               # Download, pré-processamento e split estratificado fixo (45k / 5k / 10k)
│   ├── models_custom.py         # Arquitetura da CNN Baseline (ID 0) e suas 3 variações (IDs 1, 2, 3)
│   ├── models_transfer.py       # Carregamento das redes da ImageNet e regras de congelamento (IDs 4 a 7)
│   └── engine.py                # Motor único de treino, validação, cronômetro e escrita do JSON/pesos
│
├── run_project.py               # Ponto de entrada 1: Orquestrador que recebe a lista de IDs para treinar
├── evaluate.py                  # Ponto de entrada 2: Lê a pasta results/, gera gráficos e roda o Teste (10k)
├── requirements.txt             # Dependências do projeto (torch, torchvision, matplotlib, scikit-learn, etc.)
└── README.md                    # Documentação da arquitetura e instruções de uso
```

### Papel de cada módulo
* **`src/dataset.py`:** Baixa o CIFAR-10, fixa a semente aleatória (`seed`) e aplica a divisão estratificada em $45.000$ imagens para treino e $5.000$ para validação, mantendo as $10.000$ de teste isoladas.
* **`src/models_custom.py`:** Recebe o ID (`0`, `1`, `2` ou `3`) e retorna a instância da rede neural autoral correspondente (Baseline ou uma das 3 modificações controladas).
* **`src/models_transfer.py`:** Recebe o ID (`4`, `5`, `6` ou `7`), baixa a arquitetura pré-treinada na ImageNet, substitui a camada classificadora final para $10$ saídas e configura os gradientes conforme a estratégia (congelar *backbone* vs. descongelar parcial/total).
* **`src/engine.py`:** Motor agnóstico à arquitetura. Recebe um `model`, os `dataloaders` e o dicionário de `hyperparameters`. Executa as épocas de treino e validação, salva os pesos (`.pth`) da época com maior acurácia de validação, cronometra o tempo de treino e inferência, contabiliza parâmetros totais/treináveis e grava o arquivo `.json` no disco.
* **`run_project.py`:** Define o dicionário de hiperparâmetros fixos (Loss, Otimizador, Learning Rate, Batch Size, Épocas) e percorre os IDs solicitados pelo usuário, conectando o `dataset.py`, os arquivos de `models` e o `engine.py`.
* **`evaluate.py`:** Script de análise e geração de gráficos. Opera em dois modos:
  1. *Modo Validação (`val`):* Lê todos os `.json` da pasta `results/logs/`, plota as curvas de perda e acurácia em formato vetorial (`.pdf`/`.eps`) e exibe o ranking de validação.
  2. *Modo Teste Final (`test`):* Recebe os 3 IDs vencedores (1 da Parte 1 + 2 da Parte 2), carrega seus arquivos `.pth`, executa a inferência nas $10.000$ imagens de teste do `dataset.py` e gera as matrizes de confusão e métricas finais.

---

## 3. Fluxograma Macro de Trabalho e Dados

O sistema opera em duas fases desacopladas (**Fase 1: Treinamento/Validação** e **Fase 2: Avaliação/Teste**) que se comunicam exclusivamente através dos arquivos persistidos na pasta `results/`.

```text
====================================================================================
                    FASE 1: TREINAMENTO E VALIDAÇÃO (IDs 0 a 7)
====================================================================================

                        +------------------------------+
                        |        run_project.py        |
                        |   (Recebe ex: IDs [0, 3, 4]) |
                        +--------------+---------------+
                                       |
         +-----------------------------+-----------------------------+
         | (1. Pede DataLoaders)       | (2. Instancia Modelo)       | (3. Envia Modelo +
         v                             v                             v     Dados + Hiperpar.)
+------------------+     +---------------------------+     +-------------------+
|    dataset.py    |     |     models_custom.py      |     |     engine.py     |
|                  |     |  (Se ID for 0, 1, 2 ou 3) |     |                   |
| - Baixa CIFAR-10 |     +---------------------------+     | - Loop de Épocas  |
| - Split Fixo:    |     |    models_transfer.py     |     | - Treina (45k)    |
|   45k Treino     |     |  (Se ID for 4, 5, 6 ou 7) |     | - Valida (5k)     |
|   5k Validação   |     +---------------------------+     | - Mede Tempos     |
|   10k Teste      |                                       +---------+---------+
+------------------+                                                 |
                                                                     | (4. Salva no Disco)
                                                                     v
                                                           +-------------------+
                                                           |  Pasta results/   |
                                                           | - exp_X.json      |
                                                           | - exp_X_best.pth  |
                                                           +---------+---------+
                                                                     |
=====================================================================|==============
                    FASE 2: ANÁLISE E TESTE FINAL                    |
=====================================================================|==============
                                                                     |
                                                                     v (5. Lê JSONs e Pesos)
                                                           +-------------------+
                                                           |    evaluate.py    |
                                                           +---------+---------+
                                                                     |
                                  +----------------------------------+----------------------------------+
                                  | (Etapa A: Seleção na Validação)                                     | (Etapa B: Teste Final nas 3 Campeãs)
                                  v                                                                     v
                    +-------------------------------+                                     +-------------------------------+
                    | - Lê os 8 JSONs               |                                     | - Puxa o Test Loader (10k)    |
                    | - Plota curvas Loss/Acc (PDF) |                                     |   do dataset.py               |
                    | - Imprime Tabela de Validação |                                     | - Carrega .pth dos 3 melhores |
                    |   para escolher os 3 campeões |                                     | - Gera Matriz de Confusão e   |
                    +-------------------------------+                                     |   Tabela Final do Artigo      |
                                                                                          +-------------------------------+
```

---

## 4. Como Rodar

### 4.1. Configuração do Ambiente

1. Clone o repositório e acesse a pasta raiz:
   ```bash
   git clone https://github.com/gabmm/projeto_vis_comput
   cd projeto_vis_comput
   ```
2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # ou: venv\Scripts\activate     # Windows

   pip install -r requirements.txt
   ```

### 4.2. Mapeamento dos Experimentos (IDs)

| ID | Etapa | Descrição do Experimento |
| :-: | :--- | :--- |
| **`0`** | Parte 1 (CNN Própria) | Modelo **Baseline** |
| **`1`** | Parte 1 (CNN Própria) | Modificação Arquitetural 1 |
| **`2`** | Parte 1 (CNN Própria) | Modificação Arquitetural 2 |
| **`3`** | Parte 1 (CNN Própria) | Modificação Arquitetural 3 |
| **`4`** | Parte 2 (Transfer Learning) | Arquitetura Pré-treinada A — Estratégia 1 (Backbone Congelado) |
| **`5`** | Parte 2 (Transfer Learning) | Arquitetura Pré-treinada A — Estratégia 2 (Fine-tuning) |
| **`6`** | Parte 2 (Transfer Learning) | Arquitetura Pré-treinada B — Estratégia 1 (Backbone Congelado) |
| **`7`** | Parte 2 (Transfer Learning) | Arquitetura Pré-treinada B — Estratégia 2 (Fine-tuning) |

### 4.3. Executando os Treinamentos (`run_project.py`)

Você pode rodar todos os experimentos de uma vez ou passar apenas os IDs desejados:

```bash
# Rodar apenas o Baseline (ID 0) e o primeiro experimento de Transfer Learning (ID 4)
python run_project.py --id 0 4

# Rodar apenas os experimentos da Parte 1 (CNN Própria)
python run_project.py --id 0 1 2 3

# Rodar toda a bateria de experimentos (IDs 0 a 7)
python run_project.py --id 0 1 2 3 4 5 6 7
```

### 4.4. Gerando Gráficos e Avaliando no Teste (`evaluate.py`)

**Passo 1 — Análise de Validação (escolha dos campeões):**  
Gera as curvas de aprendizado em PDF/EPS e exibe a tabela comparativa no conjunto de validação ($5.000$ imagens):
```bash
python evaluate.py --mode val
```

**Passo 2 — Avaliação Final no Conjunto de Teste ($10.000$ imagens):**  
Após identificar os 3 melhores IDs na tabela de validação (ex: ID `2` na Parte 1, e IDs `5` e `7` na Parte 2), execute a avaliação final no conjunto de teste:
```bash
python evaluate.py --mode test --winners 2 5 7
```

---

## 5. Esquema de Execução Passo a Passo (Trace de Exemplo)

Abaixo está o rastreamento completo do que acontece internamente quando executamos os experimentos **`[0, 4]`** no `run_project.py` e, em seguida, consolidamos os resultados no `evaluate.py`.

```text
[USUÁRIO] ──> python run_project.py --id 0 4
  │
  ├─ 1. run_project.py carrega o dicionário de hiperparâmetros fixos (LR, Batch Size, Épocas, Loss).
  │
  ├─ 2. run_project.py ──> chama src/dataset.py: get_dataloaders(batch_size, seed=42)
  │       └─ dataset.py baixa o CIFAR-10, aplica o split estratificado fixo e
  │          retorna (train_loader [45k], val_loader [5k]) para run_project.py.
  │
  ├─ 3. Loop de Experimentos — Iteração ID = 0:
  │       ├─ run_project.py ──> chama src/models_custom.py: get_custom_model(exp_id=0)
  │       │    └─ Retorna instância da CNN Baseline + nome "ID0_Baseline_CNN".
  │       └─ run_project.py ──> chama src/engine.py: train_and_validate(id=0, model, loaders, hyperparams)
  │            ├─ Conta parâmetros totais e treináveis.
  │            ├─ Treina nas 45k imagens e valida nas 5k ao longo das épocas.
  │            ├─ Salva o melhor estado em: results/weights/exp_0_best.pth
  │            ├─ Cronometra tempos e salva telemetria em: results/logs/exp_0.json
  │            └─ Retorna resumo para run_project.py.
  │
  └─ 4. Loop de Experimentos — Iteração ID = 4:
          ├─ run_project.py ──> chama src/models_transfer.py: get_transfer_model(exp_id=4)
          │    └─ Baixa rede da ImageNet, congela backbone, ajusta saída para 10 classes e retorna model.
          └─ run_project.py ──> chama src/engine.py: train_and_validate(id=4, model, loaders, hyperparams)
               ├─ Executa treino/validação usando exatamente os mesmos loaders.
               ├─ Salva o melhor estado em: results/weights/exp_4_best.pth
               ├─ Salva telemetria em: results/logs/exp_4.json
               └─ Retorna resumo para run_project.py (Fim do script).

--------------------------------------------------------------------------------------------------------

[USUÁRIO] ──> python evaluate.py --mode val
  │
  ├─ 1. evaluate.py lê todos os arquivos results/logs/exp_*.json.
  ├─ 2. Exporta curvas de Loss e Acurácia (Treino vs. Validação) em PDF para results/figures/.
  └─ 3. Imprime ranking de validação no terminal (Ex: Campeões identificados -> IDs 2, 5 e 7).

--------------------------------------------------------------------------------------------------------

[USUÁRIO] ──> python evaluate.py --mode test --winners 2 5 7
  │
  ├─ 1. evaluate.py ──> chama src/dataset.py e requisita, pela única vez, o test_loader (10k imagens).
  ├─ 2. Para cada ID vencedor (2, 5 e 7):
  │       ├─ Reconstrói a arquitetura vazia (via models_custom.py ou models_transfer.py).
  │       ├─ Carrega os pesos salvos em results/weights/exp_{id}_best.pth.
  │       └─ Roda a inferência nas 10.000 imagens de teste.
  └─ 3. Exporta para results/figures/ as 3 Matrizes de Confusão em formato vetorial (.pdf)
        e a tabela comparativa final (Acurácia no Teste, Parâmetros, Tempos de Treino e Inferência).
```

---

## 6. Ambiente de Execução dos Experimentos

Todos os experimentos oficiais (IDs `0` a `7`, 20 épocas cada) foram executados em uma única máquina, com a configuração abaixo.

| Componente | Especificação |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3070 (8 GB) |
| **CPU** | Intel Core i5-9400F (6 núcleos) |
| **RAM** | 16 GB (2 × 8 GB DDR4) |
| **Sistema Operacional** | Windows 10 Pro |
| **Python** | 3.14 |
| **PyTorch / torchvision** | 2.14.1 / 0.29.1 (CUDA 12.6) |

A semente aleatória foi fixada em `42` para todos os experimentos, incluindo a divisão estratificada entre treino ($45.000$ imagens) e validação ($5.000$ imagens).

> **Nota sobre os tempos:** os tempos de treino e de inferência registrados nos logs dependem deste hardware e incluem o carregamento dos dados pelo `DataLoader`.