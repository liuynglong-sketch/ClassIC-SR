# Evaluation Outputs

This folder stores generated evaluation outputs.

It is intentionally empty before running tests. Running `tester.py` will create subfolders such as:

```text
results/test8k_fp32/
results/test2k_bf16/
results/test8k_int8/
```

Do not commit large generated images or full evaluation outputs by default. Keep only small expected-result documentation files in the repository.
