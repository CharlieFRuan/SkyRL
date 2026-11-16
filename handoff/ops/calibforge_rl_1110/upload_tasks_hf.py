"""Upload the task sets of the CalibForge RL runs to a PRIVATE Hugging Face dataset (run on the driver).
    set -a; . $D/keys.env; set +a; HF_TOKEN=$HF_API_KEY python upload_tasks_hf.py <repo name> <file> [<file> ...]
Creates <account>/<repo name> as a private dataset if needed, refuses to upload if it is not private."""
import os, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{api.whoami()['name']}/{sys.argv[1]}"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
info = api.dataset_info(repo)
assert info.private, f"{repo} is not private; nothing uploaded"
for f in sys.argv[2:]:
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo, repo_type="dataset")
    print("uploaded", os.path.basename(f), os.path.getsize(f), flush=True)
info = api.dataset_info(repo, files_metadata=True)
print("repo", repo, "private", info.private)
for s in info.siblings:
    print("  ", s.rfilename, s.size)
