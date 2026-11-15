# CV-Leishmaniasis

Classificação binária (positivo / negativo) de imagens de microscopia para diagnóstico de **leishmaniose**, usando redes convolucionais pré-treinadas (ResNet50, DenseNet121, EfficientNet-B0).

O projeto avalia os modelos em três cenários sobre três datasets públicos:

| Cenário | Script | O que mede |
|---|---|---|
| Busca de hiperparâmetros | `optuna_search_model.py`, `average_search_model.py` | k-fold estratificado + Optuna |
| Validação k-fold para épocas | `test_kfold.py` | desempenho médio por fold, com early stopping |
| Intra-dataset | `test_model.py` | treino e teste no mesmo dataset (split fixo), 5 seeds |
| Cross-dataset | `test_crossdataset.py` | treino em um dataset (ou combinação) e teste em outro, 5 seeds |

Também há uma interface em Streamlit (`test_app.py`) para visualizar predições com **Grad-CAM**, que deve ser apontado a um modelo treinado.

```powershell
streamlit run .\src\test_app.py
```

---

## Instalação

Requer Python 3.11 e, de preferência, GPU NVIDIA com CUDA (o `requirements.txt` usa `torchvision` com CUDA 12.6).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Datasets

Os datasets não são versionados. Só existe uma cópia dos dados. A divisão entre treino/teste de cada dataset é fixa:

```
datasets/
  AIR_LEISH/{train,test}/{Negative,Positive}/
  DeepLeish/{train,test}/...
  DLB/{train,test}/...
```

Os demais conjuntos são montados em código por [`data/datasets.py`](src/data/datasets.py), a partir do nome:

| Nome | Conteúdo |
|---|---|
| `DLB` | `datasets/DLB/train` (o teste vem de `load_test_dataset("DLB")`) |
| `CROSS_DLB` | dataset completo: `train` + `test` |
| `DLB+AIR` | `CROSS_DLB` + `CROSS_AIR_LEISH` |

Os scripts em [`utils/`] ajudam a montar essa estrutura a partir dos dados brutos.

## Como rodar

Os scripts usam caminhos relativos (`../datasets`, `experiments/...`), então **devem ser executados de dentro de `src/`**:

Cada script tem uma lista `configs` no topo, com as combinações de modelo, augmentation e hiperparâmetros a rodar.

### Configuração de treino

Toda execução é descrita por um `TrainingConfig` ([`training/type/training_types.py`](src/training/type/training_types.py)):

| Campo | Valores |
|---|---|
| `model_name` | `resnet50`, `densenet121`, `efficientnetb0` |
| `fine_tuning` | `None` (só o classificador), `last_block`, `last_two_blocks` |
| `augmentation_level` | `weak`, `medium`, `strong` |
| `loss_name` | `cross_entropy` (BCE com `pos_weight` para desbalanceamento) |
| `optimizer_name` | `adamw`, `adam` |
| `scheduler_name` | `None`, `warmup+cosine` |
| `resolution` | `(largura, altura)` |
| `training_seed` | seed aplicada internamente pelas funções de treino |
| `dropout`, `learning_rate`, `weight_decay`, `batch_size`, `epochs` | — |

O k-fold usa também um `KFoldConfig` (`n_splits`, `val_split_seed`, `patience_early_stopping`, `metric_to_monitor`).

### Reprodutibilidade

- A seed vem de `TrainingConfig.training_seed` e é aplicada por `set_seed` dentro de `train_model`, `train_validate_model` e no início de cada fold do k-fold.
- `test_model.py` e `test_crossdataset.py` repetem cada configuração com as seeds `41–45` e reportam média e desvio padrão.
- O split do k-fold é estratificado por classe e por dataset de origem, com `val_split_seed` fixo.

## Saídas

Tudo é gravado em `src/experiments/`. Cada script define seus caminhos:

- **CSV de resultados:** uma linha por experimento, com a config completa, métricas agregadas (`*_mean` / `*_std` em %, `mcc` bruto, matriz de confusão somada), tempo e seeds usadas.
- **`.npz`:** `labels` e `probabilities` de cada execução/fold, em `<pasta>/<experiment_id>/`.
- **Checkpoints `.pth`** (intra e cross-dataset): nome no formato `{dataset}_{modelo}_{aug}_{hash}_{seed}.pth`, em que `hash` são 8 caracteres do MD5 da config (sem a seed). Se o checkpoint já existir, o modelo é carregado em vez de treinado de novo. Se qualquer hiperparâmetro mudar, o hash muda e o modelo é retreinado.

Métricas calculadas na validação ([`ValidationMetrics`](src/training/type/metrics_types.py)): loss, accuracy, balanced accuracy, MCC, precision, recall, specificity, F1 (classe 1, classe 0 e macro), ROC-AUC, PR-AUC (por classe) e TN/FP/FN/TP.

## Estrutura do código

```
src/
├── test_kfold.py / test_model.py / test_crossdataset.py   # scripts de experimento: configs, loop de seeds, gravação de arquivos
├── optuna_search_model.py / average_search_model.py       # busca de hiperparâmetros com Optuna
├── test_app.py                                            # app Streamlit com Grad-CAM
├── training/
│   ├── train_model.py                  # treino puro (sem validação)
│   ├── train_validate_model.py         # treino + teste (intra-dataset)
│   ├── train_validate_crossdataset.py  # treino + teste em outro dataset
│   ├── train_validate_kfold.py         # k-fold com early stopping e pruning do Optuna
│   ├── core/                           # train_one_epoch, validate_model, set_seed
│   └── type/                           # TrainingConfig, KFoldConfig, TrainMetrics, ValidationMetrics
├── data/
│   ├── spreadsheet.py                  # agrega métricas e grava os CSVs
│   ├── npz.py                          # grava labels/probabilidades
│   └── checkpoint.py                   # salvar/carregar modelos e hash da config
├── augmentation/                       # transformações weak / medium / strong
├── models/                             # construção de ResNet, DenseNet e EfficientNet binárias
├── handler/                            # factories (modelo, otimizador, loss, scheduler, augmentation)
└── training_config/                    # dataloader, pos_weight, grupos de parâmetros, scheduler
```

As funções em `training/` só treinam e avaliam. Quem lê e grava arquivos (CSV, `.npz`, checkpoints) são os scripts de experimento.

## Utilitários

Scripts avulsos de preparação dos dados, em [`utils/`](utils/). Cada um tem os caminhos configurados no topo do arquivo:

| Script | Função |
|---|---|
| `split-dataset.py` | separa um dataset em `train`/`test` (80/20, seed 42) |
| `check-duplicated-images.py` | procura imagens duplicadas por hash perceptual |
| `convert-image-resolution.py` | ajusta imagens para proporção 4:3 |
| `convert-image-to-rgb.py` | converte imagens para RGB |
| `cut-image-border.py` | remove bordas das imagens |
| `get-image-informations.py` | levanta resoluções, canais e transparência |
| `coco-masks/` | recorta e separa por classe as imagens do AIR_LEISH a partir das anotações COCO |