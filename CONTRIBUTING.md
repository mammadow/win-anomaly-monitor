# Contributing to ML System Resource Monitor (`resmon`)

Thank you for your interest in contributing! This project is an engineering-first, low-overhead system monitor and diagnostic suite designed specifically for Windows 11 power users and hybrid laptop architectures.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mammadow/win-anomaly-monitor.git
   cd win-anomaly-monitor
   ```

2. **Set up a virtual environment (Python 3.11+ recommended):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Run the test suite:**
   ```bash
   python -m pytest tests
   ```

---

## 📐 Architecture & Principles

- **Zero-impact overhead:** The monitor runs in Windows 11 EcoQoS (Efficiency Mode) on low-priority threads. Code must avoid allocating heavy memory structures in the 1 Hz loop.
- **Fast Win32/NT Native Calls:** Avoid slow shell queries or excessive WMI calls. Process sweeps utilize native `NtQuerySystemInformation` via ctypes (~5 ms per sweep of 400+ processes).
- **Graceful Concurrency:** Storage uses SQLite in WAL mode with busy timeouts so that CLI readers (`status`, `report`, `explain`) never block the background engine.
- **No Heavy External Dependencies for Diagnostics:** The diagnostics layer uses Python's standard library (`urllib.request`) to communicate with local Ollama or OpenAI endpoints, ensuring zero bloat in `requirements.txt`.

---

## 🧪 Testing Guidelines

- All detectors and rule modifications must include unit tests in `tests/`.
- Synthetic testing should leverage `SimSampler`, `SimProcs`, and mock samples via `conftest.py`.
- Run tests before submitting a PR:
  ```bash
  python -m pytest tests
  ```

---

## 📝 Pull Request Process

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/my-enhancement
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Verify that all 120+ tests pass cleanly.
4. Submit your pull request to the `main` branch.
