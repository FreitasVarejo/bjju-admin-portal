# Guia de Troubleshooting - Stage 1

## Índice
- [Problemas de Instalação](#problemas-de-instalação)
- [Problemas de Execução](#problemas-de-execução)
- [Problemas de Validação](#problemas-de-validação)
- [Problemas de Performance](#problemas-de-performance)
- [Problemas de Qualidade](#problemas-de-qualidade)
- [Análise de Logs](#análise-de-logs)
- [Debugging Avançado](#debugging-avançado)

---

## Problemas de Instalação

### Erro: ModuleNotFoundError: No module named 'cv2'

**Sintoma**:
```bash
ModuleNotFoundError: No module named 'cv2'
```

**Causa**: OpenCV não instalado corretamente.

**Solução**:
```bash
# Desinstalar versões antigas
pip uninstall opencv-python opencv-contrib-python opencv-python-headless

# Reinstalar versão correta
pip install opencv-python==4.9.0.80

# Verificar instalação
python -c "import cv2; print(cv2.__version__)"
# Deve imprimir: 4.9.0
```

**Solução Docker**:
```dockerfile
# No Dockerfile, adicione dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

---

### Erro: ImportError: libGL.so.1: cannot open shared object file

**Sintoma**:
```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Causa**: Bibliotecas do sistema necessárias para OpenCV não instaladas.

**Solução Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**Solução CentOS/RHEL**:
```bash
sudo yum install -y mesa-libGL glib2
```

---

### Erro: No module named 'yaml'

**Sintoma**:
```bash
ModuleNotFoundError: No module named 'yaml'
```

**Solução**:
```bash
pip install PyYAML==6.0.1
```

---

### Erro: loguru not found

**Sintoma**:
```bash
ModuleNotFoundError: No module named 'loguru'
```

**Solução**:
```bash
pip install loguru==0.7.2
```

---

## Problemas de Execução

### Erro: No images found to process

**Sintoma**:
```
Found 0 candidate images
Warning: No images found in /app/images
```

**Causas Possíveis**:

1. **Diretório vazio**
   ```bash
   # Verificar conteúdo
   ls -la data/raw/
   
   # Se vazio, adicione imagens
   cp /path/to/images/*.jpg data/raw/
   ```

2. **Filename não match o pattern**
   ```bash
   # Verificar filenames
   ls data/raw/
   
   # Devem seguir: YYYYMMDD{session}.jpg
   # Exemplo: 20260315_1.jpg
   
   # Renomear se necessário
   mv image_001.jpg 20260315_1.jpg
   ```

3. **Extensão incorreta**
   ```bash
   # Verificar extensões
   ls data/raw/*.jpg
   ls data/raw/*.jpeg
   
   # Converter se necessário
   for f in *.png; do convert "$f" "${f%.png}.jpg"; done
   ```

4. **Caminho incorreto no config**
   ```yaml
   # Verificar em pipeline_config.yaml
   paths:
     raw_images: "/app/images"  # Docker
     # ou
     raw_images: "data/raw"     # Local
   ```

---

### Erro: FileNotFoundError: Directory does not exist

**Sintoma**:
```
FileNotFoundError: Directory does not exist: /app/images
```

**Solução**:

1. **Criar diretório**:
   ```bash
   mkdir -p data/raw
   ```

2. **Verificar montagem Docker**:
   ```bash
   # Deve ter -v no docker run
   docker run -v $(pwd)/data/raw:/app/images ...
   
   # Verificar dentro do container
   docker exec <container_id> ls /app/images
   ```

3. **Verificar permissões**:
   ```bash
   # Dar permissão de leitura
   chmod -R 755 data/raw/
   ```

---

### Erro: PermissionError: Permission denied

**Sintoma**:
```
PermissionError: [Errno 13] Permission denied: '/app/data/preprocessed/20260315_1.jpg'
```

**Solução**:

1. **Permissões do diretório**:
   ```bash
   # Dar permissão de escrita
   chmod -R 777 data/preprocessed/
   ```

2. **Docker user**:
   ```dockerfile
   # No Dockerfile, adicione
   RUN mkdir -p /app/data && chmod -R 777 /app/data
   ```

3. **Executar como root** (não recomendado):
   ```bash
   docker run --user root ...
   ```

---

## Problemas de Validação

### Erro: Validation failed: dimensions_too_small

**Sintoma**:
```
Validation failed: 20260315_1.jpg - dimensions_too_small
  Dimensions 320x240 below minimum 640x480
```

**Causas**:
- Imagem de baixa resolução
- Screenshot ou thumbnail

**Soluções**:

1. **Reduzir requisitos** (se aceita baixa resolução):
   ```yaml
   validation:
     min_width: 320
     min_height: 240
   ```

2. **Upscale imagem** (não recomendado):
   ```bash
   convert input.jpg -resize 640x480 output.jpg
   ```

3. **Ignorar imagem**:
   - O pipeline vai pular automaticamente
   - Checar em `data/failed/`

---

### Erro: Validation failed: invalid_aspect_ratio

**Sintoma**:
```
Validation failed: 20260315_1.jpg - invalid_aspect_ratio
  Aspect ratio 4.50 outside range [0.5, 3.0]
```

**Causa**: Imagem panorâmica ou muito alongada.

**Solução**:

1. **Ampliar range**:
   ```yaml
   validation:
     aspect_ratio:
       min: 0.2
       max: 5.0
   ```

2. **Crop imagem**:
   ```bash
   # Crop para aspect ratio 16:9
   convert input.jpg -gravity center -crop 16:9 output.jpg
   ```

---

### Erro: Validation failed: corrupted_file

**Sintoma**:
```
Validation failed: 20260315_1.jpg - corrupted_file
  JPEG integrity check failed
```

**Causa**: Arquivo JPEG corrompido ou truncado.

**Diagnóstico**:
```bash
# Verificar JPEG
jpeginfo -c 20260315_1.jpg

# Tentar abrir com ImageMagick
identify 20260315_1.jpg

# Ver tamanho
ls -lh 20260315_1.jpg
```

**Soluções**:

1. **Redownload imagem** (se possível)

2. **Tentar recuperar**:
   ```bash
   # Converter para PNG e voltar para JPEG
   convert 20260315_1.jpg temp.png
   convert temp.png 20260315_1_fixed.jpg
   ```

3. **Desabilitar verificação** (não recomendado):
   ```yaml
   validation:
     verify_integrity: false
   ```

---

## Problemas de Performance

### Performance muito lenta (> 10s por imagem)

**Diagnóstico**:

1. **Habilitar profiling**:
   ```yaml
   performance:
     enable_profiling: true
   logging:
     level: "DEBUG"
   ```

2. **Verificar logs**:
   ```bash
   tail -f data/logs/*.log | grep "took"
   ```

**Soluções por Causa**:

#### Causa 1: max_dimension muito alto

```yaml
# ANTES (lento)
preprocessing:
  max_dimension: 4096  # 8s por imagem

# DEPOIS (rápido)
preprocessing:
  max_dimension: 2048  # 2s por imagem
```

#### Causa 2: Bilateral filter muito agressivo

```yaml
# ANTES (lento)
bilateral_filter:
  enabled: true
  d: 15  # Muito alto!

# DEPOIS (rápido)
bilateral_filter:
  enabled: true
  d: 9   # Valor ideal
```

#### Causa 3: CLAHE com grid muito fino

```yaml
# ANTES (lento)
clahe:
  tile_grid_size: [32, 32]  # Muito fino

# DEPOIS (rápido)
clahe:
  tile_grid_size: [8, 8]    # Padrão
```

#### Causa 4: Integrity check habilitado

```yaml
# Ganhe ~40ms por imagem
validation:
  verify_integrity: false
```

#### Causa 5: Saving intermediate steps

```yaml
# Desabilite para produção
debug:
  save_intermediate_steps: false
```

---

### Timeout Error

**Sintoma**:
```
Failed to process 20260315_1.jpg: timeout
  Processing exceeded 5.0 seconds
```

**Soluções**:

1. **Aumentar timeout**:
   ```yaml
   performance:
     max_processing_time_per_image: 10
   ```

2. **Otimizar config** (ver seção acima)

3. **Verificar recursos do sistema**:
   ```bash
   # CPU usage
   top
   
   # Memory
   free -h
   
   # Disk I/O
   iostat -x 1
   ```

---

### Memory Error

**Sintoma**:
```
MemoryError: Unable to allocate array with shape (4032, 3024, 3)
```

**Causa**: Imagem muito grande + pouca RAM.

**Soluções**:

1. **Reduzir max_dimension**:
   ```yaml
   preprocessing:
     max_dimension: 1536  # Ao invés de 2048
   ```

2. **Processar em batches menores**:
   ```yaml
   debug:
     max_images_to_process: 10  # 10 por vez
   ```

3. **Aumentar RAM** (se em VM/Container):
   ```bash
   # Docker
   docker run --memory="4g" ...
   ```

---

## Problemas de Qualidade

### Imagens saindo borradas

**Causa**: Bilateral filter muito agressivo.

**Solução**:
```yaml
bilateral_filter:
  d: 5              # Reduzir de 9
  sigma_color: 50   # Reduzir de 75
  sigma_space: 50   # Reduzir de 75
```

Ou desabilitar:
```yaml
bilateral_filter:
  enabled: false
```

---

### Imagens muito escuras/claras

**Causa**: CLAHE muito agressivo.

**Solução**:
```yaml
clahe:
  clip_limit: 1.5   # Reduzir de 2.0
```

Ou desabilitar:
```yaml
clahe:
  enabled: false
```

---

### Artefatos de compressão visíveis

**Causa**: output_quality muito baixo.

**Solução**:
```yaml
preprocessing:
  output_quality: 95  # Aumentar de 85
```

**Ou**: Bilateral filter desabilitado.

```yaml
bilateral_filter:
  enabled: true
  d: 9
```

---

### Cores alteradas

**Causa**: ensure_rgb desabilitado.

**Solução**:
```yaml
preprocessing:
  ensure_rgb: true  # SEMPRE true
```

---

## Análise de Logs

### Localizar Logs

```bash
# Últimos logs
ls -lt data/logs/*.log | head -1

# Ler log
tail -100 data/logs/ingestion_stage1_*.log

# Follow log em tempo real
tail -f data/logs/ingestion_stage1_*.log
```

---

### Interpretar Mensagens de Log

#### INFO: Progresso normal

```
INFO | Processing [5/15]: 20260315_1.jpg
INFO | Successfully processed 20260315_1.jpg in 2.34s
```
**Ação**: Nenhuma, tudo OK.

---

#### WARNING: Atenção necessária

```
WARNING | Validation failed for 20260315_1.jpg: dimensions_too_small
WARNING | Performance warning: 20260315_5.jpg took 6.50s (threshold: 5.0s)
```
**Ação**: 
- Revisar configuração de validação
- Otimizar preprocessing se muitos warnings de performance

---

#### ERROR: Falha no processamento

```
ERROR | Failed to process 20260315_1.jpg: corrupted_file - JPEG integrity check failed
ERROR | Preprocessing failed for 20260315_2.jpg: Out of memory
```
**Ação**:
- Verificar arquivo original
- Ajustar configuração ou recursos

---

### Filtrar Logs

```bash
# Apenas erros
grep "ERROR" data/logs/*.log

# Apenas warnings
grep "WARNING" data/logs/*.log

# Performance issues
grep "took" data/logs/*.log | grep -v "Success"

# Imagens específicas
grep "20260315_1.jpg" data/logs/*.log
```

---

### Estatísticas de Logs

```bash
# Contar sucessos
grep "Successfully processed" data/logs/*.log | wc -l

# Contar falhas
grep "Failed to process" data/logs/*.log | wc -l

# Taxa de sucesso
echo "scale=2; $(grep "Successfully" data/logs/*.log | wc -l) / $(grep "Processing \[" data/logs/*.log | wc -l) * 100" | bc
```

---

## Debugging Avançado

### Habilitar Debug Mode

```yaml
logging:
  level: "DEBUG"

debug:
  save_intermediate_steps: true
  intermediate_dir: "/app/data/intermediate"
  max_images_to_process: 1  # Apenas 1 imagem
```

### Analisar Intermediate Steps

```bash
# Processar 1 imagem com intermediate steps
python -m cv_pipeline.stage1_ingestion.ingestion

# Verificar outputs
ls -lh data/intermediate/

# Abrir cada etapa
eog data/intermediate/20260315_1_01_rgb.jpg &
eog data/intermediate/20260315_1_02_resized.jpg &
eog data/intermediate/20260315_1_03_bilateral.jpg &
eog data/intermediate/20260315_1_04_clahe.jpg &
```

**Análise**:
- `01_rgb.jpg`: Se cores OK
- `02_resized.jpg`: Se resize correto
- `03_bilateral.jpg`: Se bilateral removeu artifacts
- `04_clahe.jpg`: Se iluminação uniformizada

---

### Python Debugger

```python
# Adicionar breakpoint no código
import pdb; pdb.set_trace()

# Executar
python -m cv_pipeline.stage1_ingestion.ingestion

# Comandos no debugger
# n - next line
# s - step into
# c - continue
# p variable - print variable
# q - quit
```

---

### Verificar Metadata

```bash
# Abrir metadata de imagem processada
cat data/preprocessed/20260315_1_metadata.json | jq

# Verificar operações aplicadas
cat data/preprocessed/20260315_1_metadata.json | jq '.preprocessing_operations'

# Verificar tempo de processamento
cat data/preprocessed/20260315_1_metadata.json | jq '.metadata.preprocessing.processing_time_seconds'
```

---

### Batch Metadata

```bash
# Abrir batch metadata
cat data/preprocessed/batch_metadata.json | jq

# Success rate
cat data/preprocessed/batch_metadata.json | jq '.success_rate_percent'

# Tempo médio
cat data/preprocessed/batch_metadata.json | jq '.average_processing_time_seconds'

# Listar falhas
cat data/preprocessed/batch_metadata.json | jq '.failed_images[].filename'
```

---

### Testar Componentes Isoladamente

#### Test Scanner

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import create_scanner_from_config, load_config

config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
scanner = create_scanner_from_config(config)

files = scanner.scan_directory(Path("data/raw"))
print(f"Found {len(files)} files:")
for f in files:
    print(f"  - {f.name}")
```

#### Test Validator

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import create_validator_from_config, load_config

config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
validator = create_validator_from_config(config)

result = validator.validate(Path("data/raw/20260315_1.jpg"))
print(f"Valid: {result.is_valid}")
print(f"Dimensions: {result.width}x{result.height}")
print(f"Aspect ratio: {result.aspect_ratio}")
```

#### Test Preprocessor

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import create_preprocessor_from_config, load_config

config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
preprocessor = create_preprocessor_from_config(config)

result = preprocessor.preprocess(
    input_path=Path("data/raw/20260315_1.jpg"),
    output_path=Path("data/test_output.jpg"),
    save_intermediate=True,
    intermediate_dir=Path("data/debug")
)

print(f"Success: {result.success}")
print(f"Time: {result.processing_time_seconds:.2f}s")
print(f"Operations: {result.operations_applied}")
```

---

## Checklist de Troubleshooting

Quando encontrar um problema, siga este checklist:

### 1. Verificar Instalação
```bash
□ Python versão correta (3.11+)
□ Dependências instaladas (requirements.txt)
□ OpenCV funcional (import cv2)
□ Diretórios criados (data/raw, data/preprocessed, data/logs)
```

### 2. Verificar Configuração
```bash
□ pipeline_config.yaml existe
□ Sintaxe YAML válida
□ Caminhos corretos (raw_images, preprocessed_images)
□ Parâmetros razoáveis (max_dimension, timeouts)
```

### 3. Verificar Inputs
```bash
□ Imagens existem em data/raw/
□ Filenames match pattern (YYYYMMDD{session}.jpg)
□ Extensões corretas (.jpg ou .jpeg)
□ Imagens válidas (não corrompidas)
```

### 4. Verificar Logs
```bash
□ Logs sendo gerados em data/logs/
□ Nível de log apropriado (INFO ou DEBUG)
□ Erros específicos identificados
□ Performance warnings analisados
```

### 5. Verificar Outputs
```bash
□ Imagens preprocessadas em data/preprocessed/
□ Metadata JSON gerado
□ Batch metadata existe
□ Success rate aceitável (> 90%)
```

---

## Suporte e Recursos

### Comandos Úteis

```bash
# Verificar instalação completa
python -c "import cv2, PIL, yaml, loguru; print('✓ All dependencies OK')"

# Testar configuração
python -c "from cv_pipeline.stage1_ingestion import load_config; from pathlib import Path; config = load_config(Path('cv_pipeline/config/pipeline_config.yaml')); print('✓ Config OK')"

# Executar testes
pytest tests/test_stage1/ -v

# Limpar outputs
rm -rf data/preprocessed/* data/logs/* data/failed/*
```

### Logs Detalhados

```yaml
# Para máximo detalhe
logging:
  level: "DEBUG"
  console_output: true
```

### Report de Bug

Quando reportar um bug, inclua:

1. **Versão**:
   ```bash
   cat cv_pipeline/__init__.py | grep version
   ```

2. **Comando executado**:
   ```bash
   python -m cv_pipeline.stage1_ingestion.ingestion
   ```

3. **Erro completo**:
   ```bash
   # Últimas 50 linhas do log
   tail -50 data/logs/ingestion_stage1_*.log
   ```

4. **Configuração**:
   ```bash
   cat cv_pipeline/config/pipeline_config.yaml
   ```

5. **Ambiente**:
   ```bash
   python --version
   pip list | grep -E "(opencv|PIL|yaml|loguru)"
   ```

---

**Versão**: 1.0  
**Última Atualização**: 2026-03-28
