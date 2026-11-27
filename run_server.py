import sys
import uvicorn
from pathlib import Path

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

if __name__ == "__main__":
    print("=" * 70)
    print(" AI-POWERED MICROSERVICE OBSERVABILITY & ROOT CAUSE ANALYSIS SYSTEM")
    print(" Architecture: FudanSELab Train Ticket Microservices (~41 Services)")
    print(" Access Web Dashboard: http://127.0.0.1:8000")
    print(" Interactive API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
