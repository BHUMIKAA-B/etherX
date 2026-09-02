# EtherXVision: Real-Time Document Intelligence System

EtherXVision is a high-performance, FastAPI-based backend system designed for **automated document classification, field extraction, and biometric asset extraction**. It specializes in processing Indian documents with high precision using the PaddleOCR engine.

---

## 🚀 Key Features

- **Document Classification**: Automatically identifies Aadhaar, PAN, Driving Licenses, and Marksheets.
- **Intelligent Field Extraction**: Extracts critical information (Names, IDs, DOBs, Marks) with high accuracy.
- **Biometric Extraction**: Correctively crops and extracts Face images and Signatures (DL/PAN).
- **Production Ready**: Built with Gunicorn/Uvicorn, health checks, Docker support, and environment-driven configuration.
- **Performance Optimized**: Includes image downscaling, result caching, and multi-worker support.

---

## 📄 Document Support Matrix

| Document Type | Data Extracted | Biometrics |
| :--- | :--- | :--- |
| **Aadhaar** | Name, DOB, Gender, Aadhaar Number | Face |
| **PAN Card** | Name, Father's Name, PAN Number, DOB | Face, Signature |
| **Driving License** | Name, DL Number, DOB | Face, Signature |
| **Marksheet** | Board, Code, Subject Marks, Total, Result | Face |

---

## 🛠️ Architecture Overview

The system follows a modular architecture:
- `main.py`: FastAPI routes and request handling.
- `ocr_engine.py`: Single-instance PaddleOCR wrapper with resolution normalization and caching.
- `field_extractors/`: Specialized modules for document-specific parsing logic.
- `image_assets.py`: Advanced OpenCV/Haar-cascade logic for face and signature detection.
- `run_server.py`: Production-grade entry point supporting both Gunicorn (Linux) and Uvicorn (Dev).

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose

### Local Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/NexOfficialAccount/OcrModel.git
   cd OcrModel
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Copy `.env.example` to `.env` and adjust settings as needed.

5. **Run the Server**:
   ```bash
   python run_server.py
   ```

---

## 🐳 Docker Deployment

To run the system in acontainerized environment:

```bash
docker-compose up --build
```
The server will be available at `http://localhost:8000`.

---

## 📡 API Reference

### POST `/upload`
Upload an image file for processing.

**Request**: Multipart form-data with `file` field.

**Response**:
```json
{
  "document_type": "PAN",
  "fields": { "name": "...", "pan_number": "..." },
  "assets": { "face": "/outputs/face_xyz.jpg", "signature": "..." },
  "latency_ms": 1250
}
```

### GET `/health`
Returns the status of the server and OCR engine.

---

## 🧹 Maintenance

The project includes built-in cleanup policies. Session data in `uploads/` and `outputs/` is ignored by Git, but can be manually cleared. The system generates `.gitkeep` files to maintain folder structure.

