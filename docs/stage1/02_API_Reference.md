# Referência de API - Stage 1

## Índice
- [Modelos de Dados](#modelos-de-dados)
- [Scanner](#scanner)
- [Validator](#validator)
- [Preprocessor](#preprocessor)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Logger](#logger)

---

## Modelos de Dados

### ImageMetadata

Metadados completos de uma imagem processada.

```python
@dataclass
class ImageMetadata:
    """
    Estrutura de metadados para rastreamento de imagem no pipeline.
    """
    # Informações do arquivo
    original_filename: str
    original_path: Path
    file_size_bytes: int
    
    # Informações extraídas do filename
    capture_date: datetime
    session_number: int
    
    # Propriedades da imagem
    original_width: int
    original_height: int
    original_aspect_ratio: float
    
    # Informações pós-processamento
    processed_path: Optional[Path] = None
    processed_width: Optional[int] = None
    processed_height: Optional[int] = None
    
    # Rastreamento de status
    status: ImageStatus = ImageStatus.PENDING
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Histórico de operações
    preprocessing_operations: List[str] = field(default_factory=list)
    
    # Metadados extensíveis
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Métodos**:

#### `to_dict() -> Dict[str, Any]`
Converte metadados para dicionário serializável.

```python
metadata = ImageMetadata(...)
data = metadata.to_dict()
# Returns: {'original_filename': '...', 'capture_date': '2026-03-15T00:00:00', ...}
```

#### `add_preprocessing_op(operation: str) -> None`
Adiciona operação ao histórico de preprocessing.

```python
metadata.add_preprocessing_op("rgb_conversion")
metadata.add_preprocessing_op("bilateral_filter")
# metadata.preprocessing_operations = ["rgb_conversion", "bilateral_filter"]
```

#### `update_processed_dimensions(width: int, height: int) -> None`
Atualiza dimensões após preprocessing.

```python
metadata.update_processed_dimensions(2048, 1536)
# metadata.processed_width = 2048
# metadata.processed_height = 1536
```

---

### ValidationResult

Resultado detalhado da validação de imagem.

```python
@dataclass
class ValidationResult:
    """
    Resultado de validação com checagens individuais.
    """
    is_valid: bool
    filename: str
    file_path: Path
    
    # Checagens específicas
    filename_valid: bool = False
    format_valid: bool = False
    integrity_valid: bool = False
    dimensions_valid: bool = False
    aspect_ratio_valid: bool = False
    file_size_valid: bool = False
    
    # Detalhes da validação
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    file_size_mb: Optional[float] = None
    
    # Informações de erro
    failure_reason: Optional[FailureReason] = None
    error_message: Optional[str] = None
    
    # Dados extraídos (se válido)
    capture_date: Optional[datetime] = None
    session_number: Optional[int] = None
```

**Exemplo de Uso**:

```python
result = validator.validate(image_path)

if result.is_valid:
    print(f"✓ Image validated: {result.width}x{result.height}")
else:
    print(f"✗ Validation failed: {result.failure_reason.value}")
    print(f"  Reason: {result.error_message}")
```

---

### PreprocessingResult

Resultado do pipeline de preprocessing.

```python
@dataclass
class PreprocessingResult:
    """
    Resultado de preprocessing com métricas de performance.
    """
    success: bool
    input_path: Path
    output_path: Optional[Path] = None
    
    # Métricas de performance
    processing_time_seconds: float = 0.0
    operations_applied: List[str] = field(default_factory=list)
    
    # Mudanças de dimensão
    original_dimensions: tuple[int, int] = (0, 0)
    final_dimensions: tuple[int, int] = (0, 0)
    
    # Informações de erro
    error_message: Optional[str] = None
    failure_reason: Optional[FailureReason] = None
```

---

### IngestionResult

Resultado completo do processamento batch.

```python
@dataclass
class IngestionResult:
    """
    Estatísticas e resultados de processamento batch.
    """
    # Estatísticas gerais
    total_images_found: int = 0
    total_images_processed: int = 0
    total_images_failed: int = 0
    total_images_skipped: int = 0
    
    # Resultados detalhados
    successful_images: List[ImageMetadata] = field(default_factory=list)
    failed_images: List[ValidationResult] = field(default_factory=list)
    skipped_images: List[str] = field(default_factory=list)
    
    # Métricas de performance
    total_processing_time_seconds: float = 0.0
    average_processing_time_seconds: float = 0.0
    
    # Timestamp e configuração
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
```

**Métodos**:

#### `get_success_rate() -> float`
Calcula taxa de sucesso em percentual.

```python
result.total_images_found = 10
result.total_images_processed = 8
rate = result.get_success_rate()  # Returns: 80.0
```

#### `get_summary() -> str`
Gera resumo formatado.

```python
print(result.get_summary())
# Output:
# Ingestion Summary:
#   Total Found: 10
#   Processed: 8
#   Failed: 2
#   Success Rate: 80.00%
```

---

## Scanner

### FilenameParser

Extrai metadados de filenames usando regex.

```python
class FilenameParser:
    def __init__(
        self,
        pattern: str,
        case_insensitive: bool = True
    ):
        """
        Inicializa parser com padrão regex.
        
        Args:
            pattern: Regex pattern com grupos de captura
            case_insensitive: Matching case-insensitive
        """
```

**Métodos**:

#### `parse(filename: str) -> Tuple[bool, Optional[datetime], Optional[int], Optional[str]]`

Extrai data e sessão do filename.

```python
parser = FilenameParser(
    pattern=r"^(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([1-9])\.jpe?g$"
)

is_valid, date, session, error = parser.parse("20260315_1.jpg")

if is_valid:
    print(f"Date: {date}, Session: {session}")
    # Output: Date: 2026-03-15 00:00:00, Session: 1
else:
    print(f"Error: {error}")
```

**Returns**:
- `is_valid`: Se o filename é válido
- `capture_date`: Data extraída ou None
- `session_number`: Número da sessão ou None
- `error_message`: Mensagem de erro ou None

---

### ImageScanner

Descobre e valida arquivos de imagem.

```python
class ImageScanner:
    def __init__(
        self,
        extensions: List[str],
        filename_parser: FilenameParser,
        recursive: bool = False
    ):
        """
        Inicializa scanner de imagens.
        
        Args:
            extensions: Lista de extensões aceitas (ex: ['.jpg', '.jpeg'])
            filename_parser: Parser para validação de filename
            recursive: Busca recursiva em subdiretórios
        """
```

**Métodos**:

#### `scan_directory(directory: Path) -> List[Path]`

Escaneia diretório e retorna arquivos válidos.

```python
scanner = ImageScanner(
    extensions=['.jpg', '.jpeg'],
    filename_parser=parser,
    recursive=False
)

files = scanner.scan_directory(Path("/app/images"))
# Returns: [Path('/app/images/20260315_1.jpg'), ...]
```

**Performance**: O(n) onde n = número de arquivos no diretório

---

## Validator

### ImageValidator

Valida integridade e características técnicas.

```python
class ImageValidator:
    def __init__(
        self,
        min_width: int,
        min_height: int,
        aspect_ratio_min: float,
        aspect_ratio_max: float,
        max_file_size_mb: float,
        verify_integrity: bool = True
    ):
        """
        Inicializa validador de imagens.
        
        Args:
            min_width: Largura mínima em pixels
            min_height: Altura mínima em pixels
            aspect_ratio_min: Aspect ratio mínimo (width/height)
            aspect_ratio_max: Aspect ratio máximo
            max_file_size_mb: Tamanho máximo do arquivo em MB
            verify_integrity: Se deve verificar integridade JPEG
        """
```

**Métodos**:

#### `validate(file_path: Path, capture_date: Optional[datetime] = None, session_number: Optional[int] = None) -> ValidationResult`

Executa validação completa.

```python
validator = ImageValidator(
    min_width=640,
    min_height=480,
    aspect_ratio_min=0.5,
    aspect_ratio_max=3.0,
    max_file_size_mb=50,
    verify_integrity=True
)

result = validator.validate(
    file_path=Path("/app/images/20260315_1.jpg"),
    capture_date=datetime(2026, 3, 15),
    session_number=1
)

if result.is_valid:
    print(f"✓ {result.width}x{result.height}, {result.aspect_ratio:.2f}")
else:
    print(f"✗ {result.failure_reason.value}: {result.error_message}")
```

**Pipeline de Validação**:
1. File exists
2. File size
3. JPEG format (magic bytes)
4. JPEG integrity (PIL verify)
5. Dimensions
6. Aspect ratio

**Performance**: 
- Quick validation (magic bytes only): ~1ms
- Full validation (with integrity): ~50ms

#### `quick_validate(file_path: Path) -> bool`

Validação rápida (apenas formato e dimensões).

```python
is_valid = validator.quick_validate(Path("/app/images/test.jpg"))
# Returns: True ou False (sem detalhes)
```

**Performance**: ~1ms por imagem

---

## Preprocessor

### ImagePreprocessor

Pipeline de preprocessing otimizado para WhatsApp.

```python
class ImagePreprocessor:
    def __init__(
        self,
        max_dimension: int = 2048,
        resize_interpolation: str = "INTER_AREA",
        ensure_rgb: bool = True,
        bilateral_enabled: bool = True,
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        output_quality: int = 95
    ):
        """
        Inicializa preprocessador de imagens.
        
        Args:
            max_dimension: Dimensão máxima após normalização
            resize_interpolation: Método de interpolação OpenCV
            ensure_rgb: Garantir conversão para RGB
            bilateral_enabled: Habilitar filtro bilateral
            bilateral_d: Diâmetro do filtro bilateral
            bilateral_sigma_color: Sigma no espaço de cor
            bilateral_sigma_space: Sigma no espaço de coordenadas
            clahe_enabled: Habilitar CLAHE
            clahe_clip_limit: Limite de clip do CLAHE
            clahe_tile_grid_size: Tamanho do grid CLAHE
            output_quality: Qualidade JPEG de saída (0-100)
        """
```

**Métodos**:

#### `preprocess(input_path: Path, output_path: Path, save_intermediate: bool = False, intermediate_dir: Optional[Path] = None) -> PreprocessingResult`

Executa pipeline completo de preprocessing.

```python
preprocessor = ImagePreprocessor(
    max_dimension=2048,
    bilateral_enabled=True,
    clahe_enabled=True
)

result = preprocessor.preprocess(
    input_path=Path("/app/images/20260315_1.jpg"),
    output_path=Path("/app/data/preprocessed/20260315_1.jpg"),
    save_intermediate=False  # Se True, salva etapas intermediárias
)

if result.success:
    print(f"✓ Processed in {result.processing_time_seconds:.2f}s")
    print(f"  Operations: {', '.join(result.operations_applied)}")
    print(f"  Dimensions: {result.original_dimensions} → {result.final_dimensions}")
else:
    print(f"✗ Failed: {result.error_message}")
```

**Pipeline de Operações** (ordem crítica):
1. RGB conversion
2. Resolution normalization (PRIMEIRO!)
3. Bilateral filtering
4. CLAHE

**Performance**: 
- Imagem 4032x3024: ~2.4s
- Imagem 2048x1536: ~1.2s
- Imagem 1920x1080: ~0.9s

**Exemplo de Debug** (intermediate steps):

```python
result = preprocessor.preprocess(
    input_path=input_path,
    output_path=output_path,
    save_intermediate=True,
    intermediate_dir=Path("/app/data/intermediate")
)

# Salva:
# - {stem}_01_rgb.jpg
# - {stem}_02_resized.jpg
# - {stem}_03_bilateral.jpg
# - {stem}_04_clahe.jpg
```

---

## Ingestion Pipeline

### IngestionPipeline

Orquestrador principal do Stage 1.

```python
class IngestionPipeline:
    def __init__(self, config: PipelineConfig):
        """
        Inicializa pipeline de ingestão.
        
        Args:
            config: Objeto de configuração do pipeline
        """
```

**Métodos**:

#### `run() -> IngestionResult`

Executa pipeline completo.

```python
config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
pipeline = IngestionPipeline(config)

result = pipeline.run()

print(result.get_summary())
# Output:
# Ingestion Summary:
#   Total Found: 15
#   Processed: 14
#   Failed: 1
#   Success Rate: 93.33%
#   Total Time: 34.56s
```

**Fluxo de Execução**:
```
1. Carrega configuração
2. Configura logger
3. Cria componentes (scanner, validator, preprocessor)
4. Escaneia diretório
5. Para cada imagem:
   a. Parse filename
   b. Valida imagem
   c. Preprocessa
   d. Gera metadata
   e. Salva resultados
6. Calcula estatísticas
7. Salva metadata batch
8. Retorna IngestionResult
```

**Tratamento de Erros**:
- Resiliente: Uma falha não para o pipeline
- Categorização: Via `FailureReason` enum
- Logging: Detalhado via loguru
- Debug: Salva imagens falhadas em `/app/data/failed/`

---

### Funções Helper

#### `load_config(config_path: Path) -> PipelineConfig`

Carrega configuração de arquivo YAML.

```python
config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
# Returns: PipelineConfig object
```

#### `run_ingestion_pipeline(config_path: Path) -> IngestionResult`

Executa pipeline completo (entry point).

```python
result = run_ingestion_pipeline(
    Path("cv_pipeline/config/pipeline_config.yaml")
)
```

---

## Logger

### PipelineLogger

Sistema de logging estruturado.

```python
class PipelineLogger:
    def configure(
        self,
        log_path: Path,
        log_filename: str = "pipeline_{time}.log",
        level: str = "INFO",
        format_string: Optional[str] = None,
        rotation: str = "100 MB",
        retention: str = "30 days",
        compression: str = "zip",
        console_output: bool = True,
        json_logs: bool = False
    ) -> None:
        """
        Configura sistema de logging.
        
        Args:
            log_path: Diretório para logs
            log_filename: Pattern do nome do arquivo
            level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_string: String de formatação customizada
            rotation: Quando rotacionar logs
            retention: Quanto tempo manter logs
            compression: Formato de compressão
            console_output: Se deve logar no console também
            json_logs: Se deve usar formato JSON
        """
```

**Uso**:

```python
from cv_pipeline.stage1_ingestion.logger import get_logger

logger = get_logger()
logger.configure(
    log_path=Path("/app/data/logs"),
    level="INFO",
    rotation="100 MB",
    retention="30 days"
)
```

**Métodos Helper**:

```python
# Logging de processamento
logger.log_image_processing_start("20260315_1.jpg", Path("/app/images/20260315_1.jpg"))
logger.log_image_processing_success("20260315_1.jpg", 2.34, ["rgb", "bilateral"])
logger.log_image_processing_failure("20260315_1.jpg", "corrupted", "Invalid JPEG")

# Logging de validação
logger.log_validation_failure("test.jpg", "dimensions", "640x480", "320x240")

# Logging de operações
logger.log_preprocessing_operation("test.jpg", "bilateral_filter", "d=9")

# Logging de batch
logger.log_batch_start(15)
logger.log_batch_complete(15, 14, 1, 0, 34.56)

# Performance
logger.log_performance_warning("test.jpg", 6.5, 5.0)
```

---

## Factory Functions

Funções de conveniência para criar objetos a partir de configuração.

### `create_scanner_from_config(config: PipelineConfig) -> ImageScanner`

```python
scanner = create_scanner_from_config(config)
files = scanner.scan_directory(Path("/app/images"))
```

### `create_validator_from_config(config: PipelineConfig) -> ImageValidator`

```python
validator = create_validator_from_config(config)
result = validator.validate(image_path)
```

### `create_preprocessor_from_config(config: PipelineConfig) -> ImagePreprocessor`

```python
preprocessor = create_preprocessor_from_config(config)
result = preprocessor.preprocess(input_path, output_path)
```

---

## Enumerações

### ImageStatus

Estados de processamento de imagem.

```python
class ImageStatus(Enum):
    PENDING = "pending"              # Aguardando processamento
    VALID = "valid"                  # Validada
    INVALID = "invalid"              # Validação falhou
    PREPROCESSED = "preprocessed"    # Preprocessamento concluído
    FAILED = "failed"                # Processamento falhou
    SKIPPED = "skipped"              # Imagem pulada
```

### FailureReason

Razões de falha categorizadas.

```python
class FailureReason(Enum):
    INVALID_FILENAME = "invalid_filename"
    FILE_NOT_FOUND = "file_not_found"
    INVALID_FORMAT = "invalid_format"
    CORRUPTED_FILE = "corrupted_file"
    DIMENSIONS_TOO_SMALL = "dimensions_too_small"
    INVALID_ASPECT_RATIO = "invalid_aspect_ratio"
    FILE_TOO_LARGE = "file_too_large"
    PREPROCESSING_ERROR = "preprocessing_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"
```

---

## Exemplos Completos

### Exemplo 1: Pipeline Completo

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import (
    load_config,
    IngestionPipeline
)

# Carrega configuração
config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))

# Cria e executa pipeline
pipeline = IngestionPipeline(config)
result = pipeline.run()

# Exibe resultados
print(result.get_summary())

# Acessa detalhes
for img in result.successful_images:
    print(f"✓ {img.original_filename}: {img.preprocessing_operations}")

for failure in result.failed_images:
    print(f"✗ {failure.filename}: {failure.failure_reason.value}")
```

### Exemplo 2: Validação Individual

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import (
    create_validator_from_config,
    load_config
)

config = load_config(Path("cv_pipeline/config/pipeline_config.yaml"))
validator = create_validator_from_config(config)

image_path = Path("/app/images/20260315_1.jpg")
result = validator.validate(image_path)

if result.is_valid:
    print(f"✓ Dimensões: {result.width}x{result.height}")
    print(f"✓ Aspect ratio: {result.aspect_ratio:.2f}")
    print(f"✓ Tamanho: {result.file_size_mb:.2f} MB")
else:
    print(f"✗ Falha: {result.failure_reason.value}")
    print(f"  {result.error_message}")
```

### Exemplo 3: Preprocessing Customizado

```python
from pathlib import Path
from cv_pipeline.stage1_ingestion import ImagePreprocessor

preprocessor = ImagePreprocessor(
    max_dimension=1024,           # Menor para performance
    bilateral_enabled=True,
    bilateral_d=5,                # Mais rápido
    clahe_enabled=False,          # Desabilita CLAHE
    output_quality=90
)

result = preprocessor.preprocess(
    input_path=Path("/app/images/test.jpg"),
    output_path=Path("/app/output/test.jpg"),
    save_intermediate=True,
    intermediate_dir=Path("/app/debug")
)

print(f"Tempo: {result.processing_time_seconds:.2f}s")
print(f"Operações: {result.operations_applied}")
```

---

## Type Hints

Todos os módulos usam type hints completos:

```python
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

def validate(
    file_path: Path,
    capture_date: Optional[datetime] = None,
    session_number: Optional[int] = None
) -> ValidationResult:
    ...
```

Use `mypy` para verificação de tipos:

```bash
mypy cv_pipeline/stage1_ingestion/
```

---

**Versão**: 1.0  
**Última Atualização**: 2026-03-28
