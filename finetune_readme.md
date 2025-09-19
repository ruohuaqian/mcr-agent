Here are some concise **tips and notices** in Markdown style that you can keep in a README or project documentation for reproducibility and security:

---

### 🔑 Hugging Face Token

* **Never commit your personal token** to the repository.
* Users must set their own token as an environment variable, for example:

  ```bash
  export HF_TOKEN="your_token"
  ```

  or inside Python:

  ```python
  import os
  os.environ["HF_TOKEN"] = "your_token"
  ```
* The repository does **not** contain any personal tokens.

---

### ⚙️ Environment & Reproducibility

* Pin all critical libraries (e.g., `torch`, `transformers`, `ai2thor`) in `requirements.txt` or `environment.yml`.
* Fix random seeds to ensure deterministic results:

  ```python
  import random, numpy as np, torch
  random.seed(123)
  np.random.seed(123)
  torch.manual_seed(123)
  ```
* When using `torch.utils.data.DataLoader`, set `worker_init_fn` or `generator` to propagate the seed.

---

### 📂 Data Access

* The dataset is hosted on the Hugging Face Hub (e.g. `byeonghwikim/abp_dataset`).
* Download is handled via `hf_hub_download` or `datasets.load_dataset` with `revision=<commit_hash>` to guarantee exact versioning.
* Ensure sufficient storage (e.g., Colab ≈80 GB limit) or enable lazy-loading to avoid disk overflow.

---

### 🧩 AI2-THOR Integration

* Confirm the correct AI2-THOR version (e.g., `ai2thor==4.3.0`).
* When upgrading AI2-THOR, check API changes such as `SetObjectStates` or `GetInteractablePoses`, which may require code adjustments.
* Scene restoration (`restore_scene`) should clearly separate **annotation replay** and **random spawn** modes.

---

### 📝 Training & Evaluation

* All training, fine-tuning, and rollout scripts are standalone (`.ipynb`) and can be run via provided `.sh` wrappers.
* Each run auto-saves configuration and logs to the specified `exp/` folder for reproducibility.
* If `SR` (success rate) appears unexpectedly low, double-check:

  * `goto_interactable` logic and thresholds,
  * mask generation and instance ID pruning,
  * and Hugging Face data integrity.
