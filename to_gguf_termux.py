"""
Convert the nez_young nanoGPT checkpoint to GGUF - WITHOUT needing
PyTorch installed. Only needs numpy + gguf, both of which install
cleanly in Termux.

Why no torch: PyPI's torch wheels are built for glibc Linux and don't
match Termux's Android/bionic environment, so `pip install torch` fails
there outright. But a .pt file is just a zip archive containing a
pickle (data.pkl) plus raw tensor bytes (data/0, data/1, ...). This
script uses a custom pickle.Unpickler that intercepts torch's own
classes (torch._utils._rebuild_tensor_v2, torch.FloatStorage) and
replaces them with plain Python stand-ins, then reads each tensor's
raw bytes straight out of the zip with numpy - no torch needed.

For architecture notes (why arch="gpt2" not "llama", why no Conv1D
transpose, etc.) see to_gguf.py - same reasoning applies here.

Usage (in Termux):
    pkg install python numpy
    pip install gguf
    python to_gguf_termux.py
"""

import json
import pickle
import re
import zipfile

import numpy as np
from gguf import GGUFWriter

CKPT_PATH = "nez_young_best.pt"
TOKENIZER_PATH = "tokenizer.json"
OUT_PATH = "nez_young_best.gguf"


class _FakeStorage:
    __slots__ = ("key", "dtype", "numel")

    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class _FakeTensor:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset = storage, offset
        self.size, self.stride = size, stride


def _rebuild_tensor_v2(storage, offset, size, stride, *args, **kwargs):
    return _FakeTensor(storage, offset, size, stride)


class _TorchStubUnpickler(pickle.Unpickler):
    """Loads a torch checkpoint's data.pkl without requiring torch."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in (
            "FloatStorage", "HalfStorage", "DoubleStorage", "BFloat16Storage",
        ):
            return name
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unhandled global: {module}.{name}")

    def persistent_load(self, pid):
        _typ, storage_type, key, _location, numel = pid
        return _FakeStorage(key, storage_type, numel)


_DTYPE_MAP = {
    "FloatStorage": "<f4",
    "HalfStorage": "<f2",
    "DoubleStorage": "<f8",
}


def load_checkpoint(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    with z.open(f"{root}/data.pkl") as f:
        data = _TorchStubUnpickler(f).load()

    def materialize(tensor: _FakeTensor) -> np.ndarray:
        np_dtype = _DTYPE_MAP[tensor.storage.dtype]
        with z.open(f"{root}/data/{tensor.storage.key}") as f:
            raw = f.read()
        flat = np.frombuffer(raw, dtype=np_dtype)
        # standard contiguous row-major tensor -> reshape directly
        arr = flat[tensor.offset: tensor.offset + int(np.prod(tensor.size))]
        return arr.reshape(tensor.size).astype(np.float32)

    state_dict = {k: materialize(v) for k, v in data["model"].items()}
    return state_dict, data["config"]


def map_tensor_name(key: str):
    if key == "transformer.wte.weight":
        return "token_embd.weight"
    if key == "transformer.wpe.weight":
        return "position_embd.weight"
    if key == "transformer.ln_f.weight":
        return "output_norm.weight"
    if key == "lm_head.weight":
        return "output.weight"

    m = re.match(r"transformer\.h\.(\d+)\.(.+)", key)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    rest_map = {
        "ln_1.weight": "attn_norm.weight",
        "attn.c_attn.weight": "attn_qkv.weight",
        "attn.c_proj.weight": "attn_output.weight",
        "ln_2.weight": "ffn_norm.weight",
        "mlp.c_fc.weight": "ffn_up.weight",
        "mlp.c_proj.weight": "ffn_down.weight",
    }
    if rest not in rest_map:
        return None  # e.g. "attn.bias" - the causal mask buffer, not a weight
    return f"blk.{idx}.{rest_map[rest]}"


def main():
    print(f"Loading {CKPT_PATH} (no torch) ...")
    state_dict, cfg = load_checkpoint(CKPT_PATH)

    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    block_size = cfg["block_size"]
    vocab_size = cfg["vocab_size"]
    print(f"Config: n_layer={n_layer} n_head={n_head} n_embd={n_embd} "
          f"block_size={block_size} vocab_size={vocab_size}")

    with open(TOKENIZER_PATH) as f:
        tok = json.load(f)
    chars = tok["chars"]
    if len(chars) != vocab_size:
        raise ValueError(
            f"tokenizer has {len(chars)} chars but checkpoint vocab_size "
            f"is {vocab_size} - these must match, double check the files."
        )

    writer = GGUFWriter(OUT_PATH, arch="gpt2")
    writer.add_name("NezYoungNanoGPT")
    writer.add_context_length(block_size)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_feed_forward_length(4 * n_embd)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(0)  # 0 = F32

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(chars)
    writer.add_token_types([1] * len(chars))
    writer.add_token_merges([])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(0)

    skipped = []
    for key, arr in state_dict.items():
        name = map_tensor_name(key)
        if name is None:
            skipped.append(key)
            continue
        print(f"  {key:35s} {str(list(arr.shape)):14s} -> {name}")
        writer.add_tensor(name, arr)

    if skipped:
        print("Skipped (not weights, expected):", skipped)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
