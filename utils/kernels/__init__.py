"""Optional runtime kernels for SPARSE fast paths.

The modules in this package must remain optional: the main runtime should keep
working on machines without Triton or without CUDA by falling back to the
PyTorch implementations in ``utils.eval_core``.
"""
