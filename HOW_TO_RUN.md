# 🚀 Como Rodar o Pipeline Completo (Stages 1, 2, 3)

## 📋 Checklist do que você precisa

✅ **Ambiente**
- Python 3.10+ (você tem 3.13.7)
- CUDA/GPU (você tem RTX 4060)
- Dependências instaladas (`pip install -r requirements.txt`)

✅ **Modelos**
- YOLOv8-face: `models/yolov8/yolov8n-face.pt` (já existe)
- MobileSAM: `models/mobilesam/mobile_sam.pt` (já existe)

✅ **Imagens**
- Pasta: `data/images/` (você tem 20 imagens JPEG lá)
- Formato: YYYYMMDDN.jpeg (ex: 2026022401.jpeg)

---

## 🎯 Para Rodar Tudo (3 Estágios)

### Opção 1: Script Único (Recomendado)

```bash
cd /home/freitaspinhe/Desktop/side_projects/bjju-admin-portal

# Rodar todos os 3 estágios de uma vez
python run_complete_pipeline.py
```

**O que acontece:**
1. ✓ Stage 0: Converte imagens de `data/images/` para `data/raw/`
2. ✓ Stage 1: Processa e preprocessa imagens → `data/preprocessed/`
3. ✓ Stage 2: Detecta faces com YOLOv8 → `output/debug/detections/`
4. ✓ Stage 3: Segmenta faces com MobileSAM → `output/masks/{date}/session_{N}/`

---

## 📊 Estrutura de Saída

```
output/
├── masks/
│   └── 20260224/          # Data (YYYYMMDD)
│       └── session_1/
│           ├── 2026022401_face_001_abc123.png    # Face segmentada
│           ├── 2026022401_face_001_abc123.json   # Metadados
│           ├── 2026022401_face_002_def456.png
│           └── ...
│
├── debug/
│   ├── detections/        # Imagens com bounding boxes
│   │   └── 2026022401_detections.jpg
│   └── segmentations/     # Overlays de máscaras
│       └── 2026022401_segmentation_overlay.jpg
│
└── metadata/
    └── 20260224_session_1_manifest.json
```

---

## ⏱️ Tempo Esperado

| Stage | Tempo | Descrição |
|-------|-------|-----------|
| **Stage 0** | 1-2 min | Converte imagens para JPEG |
| **Stage 1** | 2-5 min | Preprocessa 20 imagens |
| **Stage 2** | 1-2 min | Detecta faces (YOLOv8) |
| **Stage 3** | 10-15 min | Segmenta faces (MobileSAM) |
| **Total** | ~20-25 min | Completo |

---

## 🔍 O que Você Obtém no Final

✅ **Imagens preprocessadas** em `data/preprocessed/`
- Otimizadas para detecção
- Metadata em JSON

✅ **Detecções de faces** em `output/debug/detections/`
- Imagens com bounding boxes
- Visualização das detecções

✅ **Máscaras de segmentação** em `output/masks/`
- Faces isoladas em PNG com fundo preto
- Uma por linha de detecção
- 112x112 pixels mínimo (pronto para AdaFace)

✅ **Metadados completos** em JSON
- Confiança de detecção
- Dimensões finais
- Scores de qualidade

---

## 🐛 Se Algo der Errado

### "CUDA out of memory"
```python
# Edite run_complete_pipeline.py, linha ~230:
config = SegmentationConfig(
    batch_size=4,  # Reduz de 8 para 4
    aggressive_cleanup=True
)
```

### "No images found to process"
```bash
# Verifique a pasta data/images/
ls data/images/

# Deve ter imagens no formato YYYYMMDDN.jpeg
# Exemplo: 2026022401.jpeg
```

### "Model not found"
```bash
# Verifique se os modelos existem
ls -lh models/yolov8/yolov8n-face.pt
ls -lh models/mobilesam/mobile_sam.pt

# Se não existirem, baixe:
mkdir -p models/yolov8 models/mobilesam

# YOLOv8-face
wget https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt \
  -O models/yolov8/yolov8n-face.pt

# MobileSAM
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
  -O models/mobilesam/mobile_sam.pt
```

---

## 📚 Documentação

- **README.md** - Visão geral do projeto
- **QUICKSTART.md** - Stage 1 apenas
- **QUICKSTART_STAGES_2_3.md** - Stages 2 e 3 em detalhe
- **INSTALLATION_STAGES_2_3.md** - Setup detalhado

---

## 🎬 Próximos Passos (Depois do Pipeline)

Depois que o pipeline rodar com sucesso:

1. **Revisar saída** em `output/masks/`
2. **Verificar metadados** dos arquivos JSON
3. **Integrar com AdaFace** para reconhecimento facial (Stage 4)
4. **Build da interface HITL** para revisão humana (Stage 5)

---

## ✨ Dicas

- **Primeiro teste:** Rode com 2-3 imagens antes de todo o lote
- **Debug visual:** Confira `output/debug/` para visualizar detecções e segmentações
- **Logs:** Verifique `data/logs/` para histórico detalhado
- **Reiniciar:** Stage 1 é idempotente - seguro rodar novamente

---

**Você está pronto! Rode o comando abaixo para começar:**

```bash
python /home/freitaspinhe/Desktop/side_projects/bjju-admin-portal/run_complete_pipeline.py
```
