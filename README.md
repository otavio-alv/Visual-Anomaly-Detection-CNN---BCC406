# Detecção de Anomalias Visuais — TP1 BCC406 (UFOP)

Detecção **one-class** de defeitos industriais no benchmark [VisA](https://github.com/amazon-science/spot-diff) (ECCV 2022): o modelo treina **só com imagens normais** e, no teste, atribui a cada imagem um escore de anomalia (detecção) e um mapa por pixel (localização).

**Resultado: 93,96% de AUROC de imagem** (média das 12 categorias), contra 77,3% do baseline do kit — localização: 98,61% de AUROC de pixel e 91,34% de AU-PRO@30%.

## O problema

Defeitos industriais são raros, variados e imprevisíveis, não dá para coletar exemplos de todos. O que existe em abundância são fotos de produtos **normais**. A tarefa é aprender "o que é normal" e sinalizar desvios, sem nunca ver um defeito real no treino (protocolo one-class: o teste é usado uma única vez, só na avaliação final).

## Método

**STFPM** — *Student-Teacher Feature Pyramid Matching* (Wang et al., BMVC 2021), implementado do zero em PyTorch:

- **Professora**: ResNet-18 pré-treinada no ImageNet, congelada, truncada em `layer3`.
- **Aluna**: mesma arquitetura, pesos aleatórios, treinada por retropropagação **só com imagens normais** para imitar as features da professora em 3 escalas (64×64, 32×32, 16×16), com features L2-normalizadas.
- **Intuição**: nas regiões normais a aluna imita bem; num defeito que ela nunca viu, a imitação falha. Essa discrepância, por posição, é o mapa de anomalia; o escore da imagem deriva do mapa.
- **Por que não um autoencoder?** Autoencoders de pixels tendem a generalizar demais e reconstruir o próprio defeito; comparar features de uma rede pré-treinada é mais robusto a variações normais de textura e iluminação.

### Seleção de hiperparâmetros sem tocar no teste

Como o split de validação só tem normais, criamos **anomalias sintéticas de validação** (CutPaste: um retângulo da própria imagem colado em outra posição — nunca usadas no treino). O AUROC-proxy {val normal vs. val danificada} escolhe, antes de qualquer contato com o teste:

- **data augmentation** on/off (flips aleatórios no treino — cada categoria tem duas alunas treinadas);
- **σ** da suavização gaussiana do mapa ∈ {2, 4, 8};
- **regra de pontuação** (produto das escalas + máximo / soma normalizada + máximo / produto + top-1%).

Fluxo do notebook: treinar → escolher pela validação sintética → avaliar no teste **uma única vez**.

Treino: 100 épocas, SGD (lr 0,4, momentum 0,9, wd 1e-4), batch 32, imagens 256×256, precisão mista; época final = menor perda de validação. Um modelo por categoria (~2,8M parâmetros treináveis; ~2–4 min/categoria numa T4, 33 min no total; inferência ~10 ms/imagem).

## Resultados (avaliação oficial do kit, 256×256)

| Categoria | AUROC img (%) | AUROC px (%) | AU-PRO@30% (%) |
|---|---|---|---|
| candle | 90,99 | 98,90 | 95,23 |
| capsules | 83,78 | 98,55 | 87,54 |
| cashew | 96,00 | 98,27 | 92,70 |
| chewinggum | 99,06 | 98,47 | 82,92 |
| fryum | 95,92 | 95,51 | 84,25 |
| macaroni1 | 95,04 | 99,47 | 96,60 |
| macaroni2 | 87,44 | 99,16 | 94,74 |
| pcb1 | 94,17 | 99,72 | 96,12 |
| pcb2 | 94,15 | 98,64 | 90,57 |
| pcb3 | 92,90 | 98,93 | 90,37 |
| pcb4 | 99,01 | 98,67 | 91,30 |
| pipe_fryum | 99,06 | 99,06 | 93,67 |
| **MÉDIA** | **93,96** | **98,61** | **91,34** |

Baseline do kit (ResNet-18 congelada + distância à média por posição, sem treino): 77,3% de AUROC de imagem. O notebook também traz ablação (aluna sem treino), comparação de regras de pontuação, curvas ROC e de perda, custo computacional e análise qualitativa (VP/VN/FP/FN com mapas).

## Estrutura do repositório

```
onechance.ipynb   # todo o trabalho: código + relatório + saídas
onechance.csv     # resultados oficiais por categoria (gerado pelo notebook)
kit/              # material da disciplina: enunciado (TP1.pdf), tp1_utils.py
                  # (loaders + avaliação oficial — não alterado) e baseline de exemplo
content/          # dataset VisA (não versionado — ver abaixo)
```

## Como rodar

### Google Colab (recomendado)

1. No Drive, crie `MyDrive/TP1_BCC406/` com `datasets/visa.zip` (dataset da disciplina) e `kit/tp1_utils.py`.
2. Abra `onechance.ipynb` no Colab e ative a GPU (T4).
3. Rode tudo. A primeira célula copia e descompacta o zip no disco local; checkpoints, figuras e CSV são salvos no Drive (`RETREINAR = False` reaproveita checkpoints entre sessões).

Rodada completa: ~70–80 min na T4 (2 variantes × 12 categorias). Para testar rápido, defina na célula de configuração `CATS_RODAR = ["pcb1"]` (treina/avalia só essas categorias; o CSV oficial só sai com as 12).

### Local

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn scipy pillow jupyter
export TP1_RAIZ=content          # pasta que contém visa/split.csv
jupyter notebook onechance.ipynb
```

O dataset VisA (~2 GB) não está no repositório: baixe pela disciplina ou pelo [repositório oficial](https://github.com/amazon-science/spot-diff) e adicione o arquivo .zip em `content/visa/`.

## Conformidade com as regras do TP

Só CNNs (ResNet-18); pesos pré-treinados só do ImageNet; aluna treinada do zero com retropropagação; treino e **toda** escolha de hiperparâmetros usando apenas imagens normais de `train`/`val` (+ danos sintéticos, permitidos pela regra R3); teste usado uma única vez; métricas pelo `tp1_utils.py` sem modificação; nenhuma biblioteca de detecção pronta.

## Referências

- Wang et al., *Student-Teacher Feature Pyramid Matching for Anomaly Detection*, BMVC 2021 — método.
- Zou et al., *SPot-the-Difference*, ECCV 2022 — dataset VisA.
- Li et al., *CutPaste*, CVPR 2021 — anomalias sintéticas de validação.
- Bergmann et al., *Uninformed Students*, CVPR 2020 — métrica PRO.

---
*Trabalho da disciplina BCC406 — Redes Neurais e Aprendizagem em Profundidade (DECOM/UFOP, Prof. Eduardo Luz).*
