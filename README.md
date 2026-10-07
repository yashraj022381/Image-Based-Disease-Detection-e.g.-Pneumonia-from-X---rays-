# 🩺 Image-Based-Disease-Detection-e.g.-Pneumonia-from-X---rays-
Detect medical conditions from radiology images.

**Educational AI system that classifies chest X-rays as NORMAL or PNEUMONIA, generates Grad-CAM visual explanations, and produces a safety-checked educational radiology-style report using a multi-agent RAG pipeline.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.14+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blueviolet)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Educational%20Use-yellow)](#disclaimer)


[🌐 Live Demo](#-live-demo) · [📂 Dataset](#-dataset) · [🐛 Report Bug](https://github.com/yashraj022381/Image-Based-Disease-Detection-e.g.-Pneumonia-from-X---rays-/issues)

</div>

---

## ⚠️ Important Disclaimer

> **This is an EDUCATIONAL / DEMONSTRATION project only.**  
> It is **NOT** a medical device, has **NOT** been clinically validated, and must **never** be used for real diagnosis or treatment decisions.  
> Always consult a licensed physician or radiologist for any health concern.
>
> ---

- 
> ## 🌐 Live Demo

| Service | Link |
|---------|------|
| **Streamlit Frontend** | *[Add your deployed Streamlit / Hugging Face Space URL here]* |
| **FastAPI Backend** | *[Add your deployed API URL here]* (docs at `/docs`) |

> **Tip:** You can deploy both services easily with the included `docker-compose.yml` or host the Streamlit app on [Hugging Face Spaces](https://huggingface.co/spaces) / Streamlit Community Cloud and the API on Render / Railway / Fly.io.

---

## ✨ Features

- **Binary Classification** – NORMAL vs PNEUMONIA using a fine-tuned CNN (MobileNetV2-style backbone, `stage2_final` model)
- **Grad-CAM Visual Explanations** – Heatmap overlay highlighting the regions that most influenced the prediction
- **Multi-Agent Pipeline** (LangGraph)  
  1. **Classifier Agent** – Runs the CNN  
  2. **Explainer Agent** – Generates Grad-CAM  
  3. **Report Writer Agent** – Produces a structured educational report grounded in RAG  
  4. **Safety Checker Agent** – Applies guardrails and retries if needed
- **RAG-Grounded Reports** – Report generation uses retrieved medical context (no hallucinated clinical details)
- **FastAPI Backend** – Clean REST endpoints (`/predict`, `/report`, `/agentic-report`, `/health`)
- **Streamlit Frontend** – Simple, modern UI for uploading X-rays and viewing results
- **Docker-Ready** – One-command deployment with `docker-compose`
- **Safety First** – Multiple disclaimers + automatic guardrail checks on every generated report


## 📂 Dataset

The model was trained on the well-known **Chest X-Ray Images (Pneumonia)** dataset by Kermany et al.

| Item | Link |
|------|------|
| **Kaggle Dataset** | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

**Dataset stats (approximate):**
- ~5,863 anterior-posterior chest X-rays
- Pediatric patients (1–5 years old)
- Classes: `NORMAL` / `PNEUMONIA`
- Official split: train / val / test

> The trained model (`models/stage2_final`) and config (`models/model_config.json`) are already included in this repository, so you do **not** need to retrain to run the demo.


## 🏗️ Project Structure


   pneumonia-detector
   ├── api/
   │   └── main.py                 # FastAPI application
   ├── app/
   │   └── streamlit_app.py        # Streamlit frontend
   ├── models/
   │   ├── stage2_final/           # Trained TensorFlow model
   │   └── model_config.json       # Threshold & class indices
   ├── notebooks/
   │   └── eda_and_prototyping.ipynb
   ├── src/
   │   ├── agents/
   │   │   └── pipeline_graph.py   # LangGraph multi-agent pipeline
   │   ├── genai/
   │   │   ├── guardrails.py
   │   │   ├── knowledge_base.py
   │   │   ├── rag_store.py
   │   │   └── report_chain.py     # Groq + RAG report generation
   │   └── inference.py            # Model loading, preprocessing, Grad-CAM
   ├── test_images/                # Sample X-rays for testing
   ├── Dockerfile.api
   ├── Dockerfile.streamlit
   ├── docker-compose.yml
   ├── requirements.txt
   └── README.md

🚀 Quick Start

   1. Clone the repository
      Bash
        git clone https://github.com/yashraj022381/Image-Based-Disease-Detection-e.g.-Pneumonia-from-X---rays-.git
        cd Image-Based-Disease-Detection-e.g.-Pneumonia-from-X---rays-
      
   2. Create a virtual environment & install dependencies
      Bash
        python -m venv venv
        source venv/bin/activate          # Windows: venv\Scripts\activate
        pip install -r requirements.txt

   3. Set up environment variables
      Create a .env file in the root:
     env
       GROQ_API_KEY=your_groq_api_key_here
     The Report Writer agent needs a Groq API key (free tier works fine).

   4. Run locally (two terminals)
     Terminal 1 – FastAPI backend
     Bash
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
     Terminal 2 – Streamlit frontend
     Bash
       streamlit run app/streamlit_app.py
     Open http://localhost:8501 in your browser.
     
   5. Run with Docker (recommended)
      Bash
        docker-compose up --build

   - Streamlit UI → http://localhost:8501
   - FastAPI docs → http://localhost:8000/docs


🔌 API Endpoints

   Method       Endpoint                Description
   GET          /health                 Health check
   POST         /predict                Upload X-ray → prediction + Grad-CAM
   POST         /report                 Generate educational report from prediction
   POST         /agentic-report         Full multi-agent pipeline (recommended)


Example (/agentic-report):
   Bash
     curl -X POST "http://localhost:8000/agentic-report" \
     -F "file=@test_images/sample_pneumonia.jpeg"


🧠 How the Multi-Agent Pipeline Works
     text
      Image Upload
           │
           ▼
     ┌─────────────┐
     │ Classifier  │  → CNN prediction + probability
     └──────┬──────┘
            │
            ▼
    ┌─────────────┐
    │  Explainer  │  → Grad-CAM heatmap overlay
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Report      │  → RAG-grounded educational report (Groq)
    │ Writer      │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Safety      │  → Guardrail checks (retry up to 2× if needed)
    │ Checker     │
    └──────┬──────┘
           │
           ▼
      Final Result


🛠️ Tech Stack

   Layer                 Technology
   Deep Learning         TensorFlow / Keras
   Explainability        Grad-CAM
   Backend               FastAPI + Uvicorn
   Frontend              Streamlit
   Agent Orchestration   LangGraph
   LLM + RAG             LangChain + Groq (openai/gpt-oss-120b)
   Containerization      Docker + Docker Compose

🧪 Testing

   Several test scripts are included:
   Bash
    python test_inference.py
    python test_pipeline.py
    python test_rag.py
    python test_report.py


📜 License & Citation

  - This project is released for educational and research purposes only.
  - If you use the underlying dataset, please cite:
    bibtex
      @article
      {
          kermany2018identifying,
          title={Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning},
          author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
          journal={Cell},
          volume={172},
          number={5},
          pages={1122--1131},
          year={2018},
          publisher={Elsevier}
      }

🤝 Contributing

   Pull requests, issues, and suggestions are welcome!

  Feel free to open an issue if you find a bug or have an idea for improvement.
