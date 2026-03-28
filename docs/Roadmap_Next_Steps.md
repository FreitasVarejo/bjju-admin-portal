# Roadmap & Next Steps

This document outlines the subsequent phases of the Jiu-Jitsu Attendance CV Pipeline beyond the initial prototype scope. These phases are NOT part of the current implementation but are documented here for planning continuity and architectural foresight.

---

## 1. Phase Overview

```
+-----------------------------------------------------------------------------------+
|                         FULL PIPELINE ROADMAP                                      |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|   PHASE 1 (CURRENT)              PHASE 2                PHASE 3                   |
|   +-----------------+            +-----------------+     +-----------------+       |
|   | Data Ingestion  |            | Human-in-Loop   |     | Face Embeddings |       |
|   | Face Detection  |   ---->    | Interface       |     | (AdaFace)       |       |
|   | Segmentation    |            | Quality Control |     | Storage         |       |
|   +-----------------+            +-----------------+     +-----------------+       |
|          |                              |                       |                  |
|          |                              |                       |                  |
|          v                              v                       v                  |
|   Segmented face masks           Verified & corrected     Face embedding          |
|   with metadata                  face crops               vectors                  |
|                                                                                    |
|                                                                                    |
|   PHASE 4                        PHASE 5                                          |
|   +-----------------+            +-----------------+                               |
|   | Identity        |            | Attendance      |                               |
|   | Clustering      |   ---->    | Integration     |                               |
|   | & Matching      |            | & Reporting     |                               |
|   +-----------------+            +-----------------+                               |
|          |                              |                                          |
|          v                              v                                          |
|   Identity clusters              Automated attendance                              |
|   known/unknown faces            tracking system                                   |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Phase 2: Human-in-the-Loop (HITL) Interface

### 2.1 Objective

Create a web-based interface for human operators to:
1. Review automatically segmented faces
2. Correct segmentation errors (occlusions, missed hair)
3. Flag and handle edge cases
4. Provide ground truth for potential model fine-tuning

### 2.2 Key Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Face Gallery** | Grid view of all segmented faces from a session | High |
| **Quality Indicators** | Visual flags for low confidence, possible occlusion | High |
| **Mask Editor** | Interactive tool to refine segmentation boundaries | High |
| **Accept/Reject** | Quick actions to approve or discard faces | High |
| **Batch Operations** | Process multiple similar corrections at once | Medium |
| **Annotation Export** | Export corrections as training data | Medium |

### 2.3 Technical Approach

```
+-----------------------------------------------------------------------------------+
|                          HITL INTERFACE ARCHITECTURE                               |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|   +-------------------+     +--------------------+     +-------------------+       |
|   |    FRONTEND       |     |     BACKEND        |     |     STORAGE       |       |
|   |   (React/Vue)     |<--->|    (FastAPI)       |<--->|   (PostgreSQL)    |       |
|   +-------------------+     +--------------------+     +-------------------+       |
|          |                          |                          |                   |
|          v                          v                          v                   |
|   - Face gallery           - API endpoints            - Session metadata           |
|   - Canvas mask editor     - Image serving            - Correction history         |
|   - Keyboard shortcuts     - Correction persistence   - Quality metrics            |
|   - Mobile responsive      - WebSocket updates        - Annotation storage         |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

**Frontend Stack Options**:
- React + TailwindCSS (rapid development)
- Vue 3 + Vuetify (component library)
- Canvas-based mask editing (Konva.js or Fabric.js)

**Backend Stack**:
- FastAPI (Python, async, integrates with CV pipeline)
- Image serving via static files or S3-compatible storage
- WebSocket for real-time collaboration (optional)

### 2.4 Workflow

```
1. Session Upload Complete
         |
         v
2. Auto-generate quality scores for each face
         |
         v
3. Sort faces by quality score (worst first)
         |
         v
4. HITL Operator reviews flagged faces
         |
    +----+----+
    |         |
    v         v
5a. Accept  5b. Edit mask / Reject
    |              |
    v              v
6. Mark as verified in database
         |
         v
7. Export verified faces for embedding generation
```

### 2.5 Quality Scoring Criteria

| Criterion | Weight | Threshold for Flag |
|-----------|--------|-------------------|
| SAM confidence score | 30% | < 0.85 |
| Face detection confidence | 25% | < 0.7 |
| Mask coverage ratio | 20% | < 0.6 or > 0.95 |
| Aspect ratio anomaly | 15% | Outside 0.6-1.4 |
| Resolution (min dimension) | 10% | < 80px |

---

## 3. Phase 3: Face Embedding Generation (AdaFace)

### 3.1 Objective

Generate high-quality face embedding vectors from verified segmented faces using AdaFace, suitable for downstream identity matching and clustering.

### 3.2 AdaFace Overview

**Why AdaFace**:
- State-of-the-art performance on low-quality face recognition
- Adaptive margin loss handles varying image qualities
- Robust to pose, lighting, and occlusion variations
- Pre-trained models available (WebFace4M, MS1MV2)

**Model Specifications**:
| Property | Value |
|----------|-------|
| Architecture | IR-101 (ResNet-based) |
| Embedding Dimension | 512 |
| Input Size | 112×112 |
| VRAM Usage | ~2GB (FP16) |
| Inference Speed | ~5ms/face |

### 3.3 Embedding Pipeline

```
+-----------------------------------------------------------------------------------+
|                        EMBEDDING GENERATION PIPELINE                               |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|   VERIFIED FACE                                                                    |
|   (from HITL)                                                                      |
|        |                                                                           |
|        v                                                                           |
|   +------------------+                                                             |
|   | Face Alignment   |  <-- 5-point landmark detection                            |
|   | & Normalization  |      (if not already aligned)                              |
|   +------------------+                                                             |
|        |                                                                           |
|        v                                                                           |
|   +------------------+                                                             |
|   | Resize to 112x112|  <-- Bilinear interpolation                                |
|   +------------------+                                                             |
|        |                                                                           |
|        v                                                                           |
|   +------------------+                                                             |
|   | AdaFace Inference|  <-- Forward pass through IR-101                           |
|   +------------------+                                                             |
|        |                                                                           |
|        v                                                                           |
|   +------------------+                                                             |
|   | L2 Normalize     |  <-- Unit vector embedding                                 |
|   +------------------+                                                             |
|        |                                                                           |
|        v                                                                           |
|   512-D EMBEDDING VECTOR                                                           |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

### 3.4 Face Alignment Strategy

AdaFace expects aligned faces. Use landmarks from detection stage or re-detect:

```
Alignment Transform:
1. Detect 5 facial landmarks (eyes, nose, mouth corners)
2. Compute similarity transform to canonical positions
3. Apply affine transformation
4. Crop to 112×112

Canonical Landmark Positions (112×112):
- Left Eye:  (38.29, 51.69)
- Right Eye: (73.53, 51.69)
- Nose:      (56.02, 71.73)
- Left Mouth: (41.54, 92.37)
- Right Mouth: (70.73, 92.37)
```

### 3.5 Storage Strategy

**Embedding Storage Options**:

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **PostgreSQL + pgvector** | SQL queries, ACID, familiar | Scale limits | **Phase 3** |
| Pinecone | Managed, scalable | Cost, vendor lock-in | Phase 5+ |
| Milvus | Open-source, scalable | Operational complexity | Phase 5+ |
| FAISS (local) | Fast, no infra | Not persistent | Prototyping |

**Schema Design (PostgreSQL)**:

```sql
-- Face embeddings table
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Source reference
    session_date DATE NOT NULL,
    session_number INT NOT NULL,
    source_image VARCHAR(255) NOT NULL,
    face_index INT NOT NULL,
    
    -- Embedding vector (requires pgvector extension)
    embedding vector(512) NOT NULL,
    
    -- Quality metrics
    detection_confidence FLOAT,
    segmentation_score FLOAT,
    alignment_quality FLOAT,
    
    -- Identity (populated after clustering/matching)
    identity_id UUID REFERENCES identities(id),
    identity_confidence FLOAT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_by_hitl BOOLEAN DEFAULT FALSE,
    
    -- Indexes
    CONSTRAINT unique_face UNIQUE (source_image, face_index)
);

-- Create vector similarity index
CREATE INDEX ON face_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Identities table (known individuals)
CREATE TABLE identities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    belt_rank VARCHAR(50),
    enrollment_date DATE,
    active BOOLEAN DEFAULT TRUE,
    reference_embedding vector(512),  -- Canonical embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Phase 4: Identity Clustering & Matching

### 4.1 Objective

Group face embeddings into identity clusters and match them against known individuals in the database.

### 4.2 Clustering Strategy

**Two-Stage Approach**:

```
Stage 1: Agglomerative Clustering (per-session)
- Cluster faces within a single session
- High threshold (cosine distance < 0.3)
- Groups duplicate detections of same person

Stage 2: Cross-Session Matching
- Compare session clusters to known identities
- Threshold: cosine distance < 0.5
- Create new identity if no match found
```

### 4.3 Matching Algorithm

```
FUNCTION match_face_to_identity(
    face_embedding: np.ndarray,
    known_identities: List[Identity],
    threshold: float = 0.5
) -> Tuple[Optional[Identity], float]:
    """
    Find the best matching identity for a face embedding.
    """
    best_match = None
    best_score = float('inf')
    
    FOR identity IN known_identities:
        distance = cosine_distance(face_embedding, identity.reference_embedding)
        
        IF distance < best_score:
            best_score = distance
            best_match = identity
    
    IF best_score < threshold:
        RETURN best_match, 1 - best_score  # Convert to similarity
    ELSE:
        RETURN None, 0.0  # No match found
```

### 4.4 Handling Unknown Faces

```
FUNCTION process_unknown_face(
    face_embedding: np.ndarray,
    session_metadata: Dict
) -> str:
    """
    Handle faces that don't match any known identity.
    """
    # Option 1: Create provisional identity
    provisional_id = create_provisional_identity(
        embedding=face_embedding,
        first_seen=session_metadata['date']
    )
    
    # Option 2: Queue for HITL review
    queue_for_identity_review(face_embedding, session_metadata)
    
    RETURN provisional_id
```

### 4.5 Identity Management Features

| Feature | Description |
|---------|-------------|
| **Identity Merge** | Combine two identities discovered to be same person |
| **Identity Split** | Separate mistakenly merged identities |
| **Reference Update** | Update canonical embedding with new high-quality sample |
| **Alias Support** | Handle name changes, nicknames |
| **Enrollment Flow** | UI for registering new members with photo |

---

## 5. Phase 5: Attendance Integration & Reporting

### 5.1 Objective

Connect the CV pipeline to the attendance management system, providing automated tracking and reporting.

### 5.2 Attendance Record Schema

```sql
CREATE TABLE attendance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Session info
    session_date DATE NOT NULL,
    session_number INT NOT NULL,
    session_type VARCHAR(50),  -- 'morning', 'evening', 'open_mat'
    
    -- Identity
    identity_id UUID REFERENCES identities(id),
    
    -- Detection details
    face_embedding_id UUID REFERENCES face_embeddings(id),
    match_confidence FLOAT,
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verified_by UUID,  -- Admin user
    verified_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_attendance UNIQUE (session_date, session_number, identity_id)
);
```

### 5.3 Reporting Features

| Report | Description | Frequency |
|--------|-------------|-----------|
| **Daily Attendance** | List of attendees per session | Per session |
| **Weekly Summary** | Attendance counts, trends | Weekly |
| **Monthly Report** | Detailed analytics, retention | Monthly |
| **Individual History** | Training frequency per member | On demand |
| **Belt Progress** | Attendance correlation with rank | Quarterly |

### 5.4 Integration Points

```
+-----------------------------------------------------------------------------------+
|                        SYSTEM INTEGRATION ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|   +-------------------+     +--------------------+     +-------------------+       |
|   |   CV Pipeline     |     |   Attendance API   |     |   Admin Portal    |       |
|   |   (Python)        |---->|   (FastAPI)        |<--->|   (React)         |       |
|   +-------------------+     +--------------------+     +-------------------+       |
|                                    |                                               |
|                                    v                                               |
|                            +--------------------+                                  |
|                            |   External Systems |                                  |
|                            +--------------------+                                  |
|                                    |                                               |
|                   +----------------+----------------+                              |
|                   |                |                |                              |
|                   v                v                v                              |
|            +----------+     +-----------+    +-------------+                       |
|            | Billing  |     | Mobile    |    | Notification|                       |
|            | System   |     | App       |    | Service     |                       |
|            +----------+     +-----------+    +-------------+                       |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 6. Technical Debt & Future Improvements

### 6.1 Model Improvements

| Improvement | Effort | Impact | Phase |
|-------------|--------|--------|-------|
| Fine-tune YOLOv8-face on BJJ photos | Medium | High | 4+ |
| Train custom SAM adapter for gi/hair | High | Medium | 5+ |
| Ensemble detection (YOLO + RetinaFace) | Low | Low | 3+ |
| Quality-aware embedding weighting | Medium | Medium | 4+ |

### 6.2 Infrastructure Improvements

| Improvement | Effort | Impact | Phase |
|-------------|--------|--------|-------|
| GPU queue for batch processing | Medium | High | 3+ |
| S3-compatible storage for images | Low | Medium | 2+ |
| Redis caching for embeddings | Low | Medium | 4+ |
| Kubernetes deployment | High | High | 5+ |

### 6.3 UX Improvements

| Improvement | Effort | Impact | Phase |
|-------------|--------|--------|-------|
| Mobile-friendly HITL interface | Medium | High | 2+ |
| Drag-and-drop photo upload | Low | Medium | 2+ |
| Real-time processing feedback | Medium | Medium | 3+ |
| Member self-service portal | High | Medium | 5+ |

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AdaFace fails on gi-obscured faces | Medium | High | HITL fallback, multiple photo angles |
| Clustering creates false merges | Medium | High | Conservative thresholds, HITL review |
| VRAM constraints on larger images | Low | Medium | Tiling, model optimization |
| WhatsApp compression too severe | Low | Medium | Request higher-quality uploads |

### 7.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Manual HITL bottleneck | High | Medium | Batch operations, quality prioritization |
| Data privacy concerns | Medium | High | Clear consent, data retention policy |
| System downtime during class | Low | High | Offline processing, manual fallback |

---

## 8. Resource Estimates

### 8.1 Development Effort

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 2: HITL Interface | 3-4 weeks | Phase 1 complete |
| Phase 3: AdaFace Integration | 2-3 weeks | Phase 1 complete |
| Phase 4: Clustering & Matching | 2-3 weeks | Phases 2, 3 |
| Phase 5: Attendance Integration | 4-6 weeks | Phase 4 |
| **Total** | **11-16 weeks** | |

### 8.2 Infrastructure Requirements

| Resource | Phase 1 | Phase 2-3 | Phase 4-5 |
|----------|---------|-----------|-----------|
| GPU | Local RTX 4060 | Same | Cloud GPU (optional) |
| Storage | 50GB | 200GB | 500GB+ |
| Database | SQLite/JSON | PostgreSQL | PostgreSQL + pgvector |
| Hosting | Local Docker | VPS | VPS + CDN |

---

## 9. Success Metrics

### 9.1 Phase 1 (Current) Success Criteria

- [ ] Process 95% of input images without errors
- [ ] Detect 90%+ of visible faces in group photos
- [ ] Generate usable segmentation masks for 85%+ of detections
- [ ] Meet 2-minute per image processing constraint
- [ ] No OOM errors on 8GB VRAM

### 9.2 Phase 2-5 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Face recognition accuracy | > 95% | Verified matches / Total matches |
| HITL correction rate | < 10% | Corrections needed / Total faces |
| Attendance capture rate | > 90% | Detected attendees / Actual attendees |
| False identity matches | < 2% | Incorrect matches / Total matches |
| Processing latency | < 5 min/session | End-to-end time |

---

## 10. Documentation Roadmap

As each phase is implemented, create corresponding documentation:

| Document | Phase | Status |
|----------|-------|--------|
| `Project_Overview.md` | 1 | ✅ Complete |
| `Environment_Setup.md` | 1 | ✅ Complete |
| `01_Data_Ingestion_Plan.md` | 1 | ✅ Complete |
| `02_Detection_Stage_Plan.md` | 1 | ✅ Complete |
| `03_Segmentation_Stage_Plan.md` | 1 | ✅ Complete |
| `Roadmap_Next_Steps.md` | 1 | ✅ Complete |
| `04_HITL_Interface_Plan.md` | 2 | 📝 Planned |
| `05_AdaFace_Integration_Plan.md` | 3 | 📝 Planned |
| `06_Clustering_Matching_Plan.md` | 4 | 📝 Planned |
| `07_Attendance_Integration_Plan.md` | 5 | 📝 Planned |
| `API_Reference.md` | 2+ | 📝 Planned |
| `Deployment_Guide.md` | 2+ | 📝 Planned |

---

*Document Version: 1.0*  
*Last Updated: 2026-03-28*  
*Author: CV Pipeline Planning*
