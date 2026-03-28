# Guia de Configuração - Stage 1

## Visão Geral

O Stage 1 é altamente configurável através do arquivo `pipeline_config.yaml`. Este guia detalha todas as opções disponíveis e como ajustá-las para diferentes cenários.

## Localização

```
cv_pipeline/config/pipeline_config.yaml
```

## Estrutura do Arquivo

O arquivo de configuração é dividido em 10 seções principais:

```yaml
paths:              # Configuração de diretórios
parsing:            # Regras de parsing de filename
validation:         # Critérios de validação
preprocessing:      # Parâmetros de processamento
performance:        # Configurações de performance
logging:            # Sistema de logs
metadata:           # Geração de metadados
error_handling:     # Tratamento de erros
debug:              # Modo debug
```

---

## Seção: paths

Configuração de diretórios de entrada e saída.

```yaml
paths:
  # Diretório de entrada (imagens brutas do WhatsApp)
  raw_images: "/app/images"
  
  # Diretório de saída (imagens preprocessadas)
  preprocessed_images: "/app/data/preprocessed"
  
  # Diretório de logs
  logs: "/app/data/logs"
  
  # Diretório para imagens que falharam (debug)
  failed_images: "/app/data/failed"
```

### Uso Docker

Para ambiente Docker, use caminhos absolutos dentro do container:

```yaml
paths:
  raw_images: "/app/images"           # Montado via -v
  preprocessed_images: "/app/data/preprocessed"
  logs: "/app/data/logs"
  failed_images: "/app/data/failed"
```

### Uso Local

Para desenvolvimento local, use caminhos relativos ou absolutos:

```yaml
paths:
  raw_images: "data/raw"              # Relativo ao projeto
  preprocessed_images: "data/preprocessed"
  logs: "data/logs"
  failed_images: "data/failed"
```

**Nota**: O pipeline cria diretórios automaticamente se não existirem.

---

## Seção: parsing

Regras para descoberta e parsing de arquivos.

```yaml
parsing:
  # Regex para extrair data e sessão do filename
  # Formato: YYYYMMDD{session}.jpg
  filename_pattern: "^(\\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])([1-9])\\.jpe?g$"
  
  # Extensões aceitas
  extensions:
    - ".jpg"
    - ".jpeg"
  
  # Matching case-insensitive
  case_insensitive: true
  
  # Busca recursiva (NÃO recomendado para performance)
  recursive: false
```

### Detalhamento do Regex

```regex
^                          # Início da string
(\d{4})                    # Grupo 1: Ano (4 dígitos)
(0[1-9]|1[0-2])           # Grupo 2: Mês (01-12)
(0[1-9]|[12]\d|3[01])     # Grupo 3: Dia (01-31)
([1-9])                    # Grupo 4: Sessão (1-9)
\.jpe?g                    # Extensão .jpg ou .jpeg
$                          # Fim da string
```

### Exemplos de Filenames Válidos

```
✓ 20260315_1.jpg       → 2026-03-15, sessão 1
✓ 20260315_2.jpeg      → 2026-03-15, sessão 2
✓ 20261231_9.jpg       → 2026-12-31, sessão 9
```

### Exemplos de Filenames Inválidos

```
✗ 20260315_0.jpg       → Sessão 0 inválida
✗ 20260315_10.jpg      → Sessão > 9
✗ 20260230_1.jpg       → Data inválida
✗ image_001.jpg        → Não match pattern
✗ 20260315.jpg         → Faltando sessão
```

### Customizando o Pattern

**Permitir sessões 10-99**:
```yaml
filename_pattern: "^(\\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])([1-9]\\d?)\\.jpe?g$"
```

**Adicionar PNG**:
```yaml
extensions:
  - ".jpg"
  - ".jpeg"
  - ".png"
```

---

## Seção: validation

Critérios para validação de imagens.

```yaml
validation:
  # Dimensões mínimas (largura x altura)
  min_width: 640
  min_height: 480
  
  # Constraints de aspect ratio (width/height)
  aspect_ratio:
    min: 0.5    # 1:2 (vertical)
    max: 3.0    # 3:1 (panorâmica)
  
  # Tamanho máximo do arquivo em MB
  max_file_size_mb: 50
  
  # Verificar integridade JPEG antes de processar
  verify_integrity: true
```

### Ajustando Dimensões Mínimas

**Para fotos de baixa resolução** (celulares antigos):
```yaml
min_width: 480
min_height: 320
```

**Para fotos de alta qualidade** (câmeras profissionais):
```yaml
min_width: 1920
min_height: 1080
```

### Ajustando Aspect Ratio

**Aspect Ratios Comuns**:

| Tipo | Ratio | Valor | Config |
|------|-------|-------|--------|
| Quadrado | 1:1 | 1.0 | `min: 0.9, max: 1.1` |
| Foto vertical | 3:4 | 0.75 | `min: 0.7, max: 0.8` |
| Foto 4:3 | 4:3 | 1.33 | `min: 1.2, max: 1.4` |
| HD 16:9 | 16:9 | 1.78 | `min: 1.7, max: 1.9` |
| Panorâmica | 21:9 | 2.33 | `min: 2.2, max: 2.5` |

**Para aceitar apenas landscape**:
```yaml
aspect_ratio:
  min: 1.0    # Apenas >= 1:1
  max: 3.0
```

**Para aceitar qualquer formato**:
```yaml
aspect_ratio:
  min: 0.1    # Praticamente sem restrição
  max: 10.0
```

### Integridade JPEG

**Desabilitar para performance** (se confia na fonte):
```yaml
verify_integrity: false
```
- Ganha: ~40ms por imagem
- Risco: Imagens corrompidas podem causar falha no preprocessing

**Habilitar para produção**:
```yaml
verify_integrity: true
```
- Custo: ~40ms por imagem
- Benefício: Detecta JPEGs corrompidos antes de processar

---

## Seção: preprocessing

Parâmetros do pipeline de processamento.

```yaml
preprocessing:
  # Normalização de resolução
  # Limita a dimensão maior (mantém aspect ratio)
  max_dimension: 2048
  
  # Método de interpolação para resize
  # Opções: INTER_AREA, INTER_CUBIC, INTER_LINEAR, INTER_LANCZOS4
  resize_interpolation: "INTER_AREA"
  
  # Garantir conversão para RGB
  ensure_rgb: true
  
  # Filtro bilateral (redução de artefatos JPEG)
  bilateral_filter:
    enabled: true
    d: 9                # Diâmetro (3, 5, 7, 9)
    sigma_color: 75     # Sigma no espaço de cor
    sigma_space: 75     # Sigma no espaço de coordenadas
  
  # CLAHE (uniformização de iluminação)
  clahe:
    enabled: true
    clip_limit: 2.0     # Limite de contraste (1.0-4.0)
    tile_grid_size: [8, 8]  # Tamanho do grid
  
  # Qualidade de saída JPEG
  output_format: "JPEG"
  output_quality: 95  # 0-100 (95 = alta qualidade)
```

### Ajustando max_dimension

**Impacto na Performance**:

| max_dimension | Tempo/Imagem | Uso de Memória | Qualidade |
|---------------|--------------|----------------|-----------|
| 1024 | ~0.8s | ~5 MB | Baixa |
| 1536 | ~1.2s | ~10 MB | Média |
| 2048 | ~2.0s | ~15 MB | Alta |
| 3072 | ~4.5s | ~35 MB | Muito Alta |
| 4096 | ~8.0s | ~60 MB | Máxima |

**Recomendações**:
- **Desenvolvimento/Teste**: 1024 (rápido)
- **Produção**: 2048 (balanço ideal)
- **Alta qualidade**: 3072 (se tem tempo)

### Ajustando Bilateral Filter

**Parâmetro `d` (diâmetro)**:

```yaml
d: 5   # Rápido, menos blur (~0.5s)
d: 7   # Balanceado (~0.8s)
d: 9   # Padrão, boa qualidade (~1.2s)
d: 11  # Lento, muito blur (~2.0s)
```

**Parâmetros sigma**:

```yaml
# Mais agressivo (remove mais artefatos, pode borrar)
sigma_color: 100
sigma_space: 100

# Padrão (balanceado)
sigma_color: 75
sigma_space: 75

# Mais conservador (preserva mais detalhes)
sigma_color: 50
sigma_space: 50
```

**Desabilitar para performance**:
```yaml
bilateral_filter:
  enabled: false
```
- Ganha: ~1.2s por imagem
- Perde: Qualidade em imagens do WhatsApp

### Ajustando CLAHE

**Parâmetro `clip_limit`**:

```yaml
clip_limit: 1.0   # Sutil
clip_limit: 2.0   # Padrão (recomendado)
clip_limit: 3.0   # Agressivo
clip_limit: 4.0   # Muito agressivo (pode criar artefatos)
```

**Parâmetro `tile_grid_size`**:

```yaml
tile_grid_size: [4, 4]    # Grid grosso (rápido)
tile_grid_size: [8, 8]    # Padrão (balanceado)
tile_grid_size: [16, 16]  # Grid fino (lento)
```

**Desabilitar se iluminação é uniforme**:
```yaml
clahe:
  enabled: false
```

### Interpolação de Resize

| Método | Velocidade | Qualidade | Uso |
|--------|------------|-----------|-----|
| INTER_NEAREST | Muito rápido | Baixa | Não recomendado |
| INTER_LINEAR | Rápido | Média | Upscaling |
| INTER_AREA | Médio | Alta | **Downsampling (recomendado)** |
| INTER_CUBIC | Lento | Alta | Upscaling |
| INTER_LANCZOS4 | Muito lento | Muito alta | Alta qualidade |

**Para imagens WhatsApp** (sempre downsampling):
```yaml
resize_interpolation: "INTER_AREA"  # RECOMENDADO
```

### Qualidade de Saída

```yaml
output_quality: 85   # Economia de espaço (~30% menor)
output_quality: 95   # Padrão (balanceado)
output_quality: 100  # Máxima qualidade (arquivo maior)
```

**Trade-off**:
- `quality: 85` → ~500KB por imagem
- `quality: 95` → ~800KB por imagem
- `quality: 100` → ~1.5MB por imagem

---

## Seção: performance

Configurações de performance e timeouts.

```yaml
performance:
  # Timeout máximo por imagem (segundos)
  max_processing_time_per_image: 5
  
  # Número de workers para processamento paralelo
  # 1 = sequencial, -1 = todos os cores
  num_workers: 1
  
  # Habilitar profiling de performance
  enable_profiling: false
```

### Timeout por Imagem

```yaml
max_processing_time_per_image: 3   # Agressivo
max_processing_time_per_image: 5   # Padrão
max_processing_time_per_image: 10  # Conservador
```

**Nota**: Se exceder timeout, a imagem é marcada como FAILED.

### Workers (Futuro)

**Atualmente não implementado** (sempre sequencial):
```yaml
num_workers: 1  # APENAS valor suportado
```

**Planejado para versão futura**:
```yaml
num_workers: 4   # 4 imagens em paralelo
num_workers: -1  # Usa todos os CPU cores
```

### Profiling

**Habilitar para análise de performance**:
```yaml
enable_profiling: true
```

Gera relatório detalhado de tempo por operação.

---

## Seção: logging

Configuração do sistema de logs.

```yaml
logging:
  # Nível de log
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
  # Formato de log
  format: "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
  
  # Logs em formato JSON (para parsers automáticos)
  json_logs: false
  
  # Rotação de logs
  rotation: "100 MB"    # Por tamanho
  # rotation: "1 day"   # Por tempo
  
  # Retenção de logs
  retention: "30 days"
  
  # Compressão de logs antigos
  compression: "zip"    # zip, gz, bz2
  
  # Nome do arquivo de log
  log_filename: "ingestion_stage1_{time}.log"
  
  # Também logar no console
  console_output: true
```

### Níveis de Log

| Nível | Uso | Output Esperado |
|-------|-----|-----------------|
| DEBUG | Desenvolvimento | MUITO verboso (cada operação) |
| INFO | Produção | Progresso do pipeline |
| WARNING | Produção (quieto) | Apenas problemas |
| ERROR | Mínimo | Apenas erros críticos |

**Desenvolvimento**:
```yaml
level: "DEBUG"
console_output: true
json_logs: false
```

**Produção**:
```yaml
level: "INFO"
console_output: false  # Apenas arquivo
json_logs: true        # Para parsing automático
```

### Rotação de Logs

**Por tamanho**:
```yaml
rotation: "50 MB"    # Rotaciona a cada 50MB
rotation: "100 MB"   # Padrão
rotation: "500 MB"   # Arquivos grandes
```

**Por tempo**:
```yaml
rotation: "1 hour"   # Rotaciona a cada hora
rotation: "1 day"    # Rotaciona diariamente
rotation: "1 week"   # Rotaciona semanalmente
```

**Misto**:
```yaml
rotation: "100 MB"   # Ou 100MB
retention: "7 days"  # Mas mantém apenas 7 dias
```

### Logs JSON

**Habilitar para integração com ferramentas**:
```yaml
json_logs: true
```

Exemplo de saída:
```json
{
  "text": "Processing image: 20260315_1.jpg",
  "record": {
    "elapsed": {"repr": "0:00:02.340000", "seconds": 2.34},
    "level": {"name": "INFO", "no": 20},
    "time": {"repr": "2026-03-28 10:30:45", "timestamp": 1743164445}
  }
}
```

---

## Seção: metadata

Configuração de geração de metadados.

```yaml
metadata:
  # Habilitar geração de metadata
  enabled: true
  
  # Formato (apenas JSON suportado)
  format: "json"
  
  # Incluir operações de preprocessing
  include_preprocessing_ops: true
  
  # Incluir resultados de validação
  include_validation_results: true
  
  # Pattern do nome do arquivo de metadata
  metadata_filename: "{original_filename}_metadata.json"
```

### Desabilitar Metadata

**Para performance** (não recomendado):
```yaml
metadata:
  enabled: false
```

Ganha: ~10ms por imagem
Perde: Rastreabilidade completa

### Customizar Filename

```yaml
# Padrão
metadata_filename: "{original_filename}_metadata.json"
# Resultado: 20260315_1_metadata.json

# Alternativas
metadata_filename: "{original_filename}.meta.json"
# Resultado: 20260315_1.meta.json

metadata_filename: "meta_{original_filename}.json"
# Resultado: meta_20260315_1.json
```

---

## Seção: error_handling

Estratégias de tratamento de erros.

```yaml
error_handling:
  # Continuar processando se uma imagem falhar
  continue_on_error: true
  
  # Salvar imagens que falharam para debug
  save_failed_images: true
  
  # Número máximo de retries
  max_retries: 2
  
  # Delay entre retries (segundos)
  retry_delay: 1
```

### Continue on Error

**Produção** (recomendado):
```yaml
continue_on_error: true
```
- Uma falha não para todo o pipeline
- Processa o máximo possível

**Desenvolvimento/Debug**:
```yaml
continue_on_error: false
```
- Para imediatamente no primeiro erro
- Útil para debugging

### Retries

```yaml
max_retries: 0   # Sem retries (falha imediata)
max_retries: 2   # Padrão (tenta 3x total)
max_retries: 5   # Agressivo
```

**Use retries para**:
- Erros de I/O transitórios
- Problemas de memória temporários

**Não use retries para**:
- Validação (não vai passar mesmo)
- Erros lógicos (sempre vai falhar)

---

## Seção: debug

Modo debug e ferramentas de diagnóstico.

```yaml
debug:
  # Salvar etapas intermediárias do preprocessing
  save_intermediate_steps: false
  
  # Diretório para outputs intermediários
  intermediate_dir: "/app/data/intermediate"
  
  # Habilitar debug visual (anotações em imagens)
  visual_debug: false
  
  # Limitar número de imagens a processar (0 = sem limite)
  max_images_to_process: 0
```

### Intermediate Steps

**Habilitar para debug de preprocessing**:
```yaml
debug:
  save_intermediate_steps: true
  intermediate_dir: "/app/data/intermediate"
```

Salva:
```
intermediate/
├── 20260315_1_01_rgb.jpg
├── 20260315_1_02_resized.jpg
├── 20260315_1_03_bilateral.jpg
└── 20260315_1_04_clahe.jpg
```

**Custo**: ~200ms + 4x espaço em disco por imagem

### Limitar Processamento

**Para testes rápidos**:
```yaml
max_images_to_process: 5  # Processa apenas 5 primeiras
```

**Produção**:
```yaml
max_images_to_process: 0  # Sem limite
```

---

## Cenários de Configuração

### Cenário 1: Desenvolvimento Local

```yaml
paths:
  raw_images: "data/raw"
  preprocessed_images: "data/preprocessed"
  logs: "data/logs"

preprocessing:
  max_dimension: 1024      # Rápido
  bilateral_filter:
    enabled: false         # Skip para velocidade

performance:
  max_processing_time_per_image: 10

logging:
  level: "DEBUG"
  console_output: true

debug:
  save_intermediate_steps: true
  max_images_to_process: 3  # Apenas 3 imagens
```

### Cenário 2: Produção Docker

```yaml
paths:
  raw_images: "/app/images"
  preprocessed_images: "/app/data/preprocessed"
  logs: "/app/data/logs"

preprocessing:
  max_dimension: 2048      # Qualidade
  bilateral_filter:
    enabled: true
  clahe:
    enabled: true

performance:
  max_processing_time_per_image: 5

logging:
  level: "INFO"
  console_output: false
  json_logs: true          # Para parsing

error_handling:
  continue_on_error: true
  save_failed_images: true

debug:
  save_intermediate_steps: false
  max_images_to_process: 0
```

### Cenário 3: Alta Performance

```yaml
preprocessing:
  max_dimension: 1536      # Menor
  bilateral_filter:
    enabled: true
    d: 5                   # Mais rápido
  clahe:
    enabled: false         # Skip

performance:
  max_processing_time_per_image: 2

validation:
  verify_integrity: false  # Skip

metadata:
  include_preprocessing_ops: false
  include_validation_results: false
```

**Performance esperada**: ~1s por imagem

### Cenário 4: Máxima Qualidade

```yaml
preprocessing:
  max_dimension: 3072      # Alta resolução
  bilateral_filter:
    enabled: true
    d: 11                  # Máximo
    sigma_color: 100
    sigma_space: 100
  clahe:
    enabled: true
    clip_limit: 2.0
  output_quality: 100      # Máxima

performance:
  max_processing_time_per_image: 15

validation:
  verify_integrity: true
  min_width: 1920
  min_height: 1080
```

**Performance esperada**: ~8s por imagem

---

## Validação de Configuração

### Verificar Sintaxe YAML

```bash
# Usando Python
python -c "import yaml; yaml.safe_load(open('cv_pipeline/config/pipeline_config.yaml'))"

# Usando yamllint (se instalado)
yamllint cv_pipeline/config/pipeline_config.yaml
```

### Testar Configuração

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import load_config

try:
    config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
    print("✓ Configuração válida")
    print(f"  Max dimension: {config.max_dimension}")
    print(f"  Bilateral enabled: {config.bilateral_filter_enabled}")
except Exception as e:
    print(f"✗ Erro na configuração: {e}")
```

---

## Troubleshooting

### Config não carrega

**Erro**: `FileNotFoundError`
```bash
# Verifique o caminho
ls cv_pipeline/config/pipeline_config.yaml
```

**Erro**: `yaml.scanner.ScannerError`
```bash
# Sintaxe YAML inválida, verifique indentação
yamllint cv_pipeline/config/pipeline_config.yaml
```

### Performance ruim

1. Reduza `max_dimension` para 1024 ou 1536
2. Desabilite `bilateral_filter`
3. Desabilite `clahe`
4. Set `verify_integrity: false`

### Muitas imagens falhando

1. Verifique `min_width` e `min_height`
2. Amplie range de `aspect_ratio`
3. Aumente `max_file_size_mb`
4. Check logs em `data/logs/`

---

**Versão**: 1.0  
**Última Atualização**: 2026-03-28
