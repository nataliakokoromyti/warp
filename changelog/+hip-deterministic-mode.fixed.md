Fix deterministic mode being silently inactive on HIP/ROCm devices. The device-side deterministic
helpers were compiled only when `__CUDA_ARCH__` was defined, which HIPRTC never defines, so generated
kernels fell back to plain atomics: consumed-return counters returned scheduling-order slots and the
counting pass no longer suppressed user-visible side effects. Kernels using `wp.config.deterministic`
now produce bit-identical results across runs on ROCm. Capturing a deterministic launch into a graph
is still unsupported on ROCm, because HIP rejects the launcher's temporary buffer allocations while a
capture is active.
