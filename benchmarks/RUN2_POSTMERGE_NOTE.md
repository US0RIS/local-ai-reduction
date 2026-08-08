# Run 2 post-merge note

After merging LARC v0.2, additional attempts were made to close the external validation gap:

- direct raw GitHub binary retrieval of a committed TinyStories-15M INT4 checkpoint;
- direct Hugging Face download of `ggml-org/tiny-llamas/stories260K.gguf` via its published Download link;
- installable plugin search for Hugging Face/model-registry/GPU/cloud-GPU access.

Results:

- the compute container cannot resolve public GitHub hosts directly;
- the Hugging Face file page is reachable and reports a 1.19 MB GGUF, but the signed Xet payload cannot be consumed by the execution sandbox;
- no installable connected GPU/model-registry plugin is available in this session.

These are environment-access failures, not LARC quality failures. L3 and L4 remain open exactly as defined in `docs/VALIDATION_GATES.md`.
