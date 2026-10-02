# TDLPP + SPPR — MATLAB source package

Download [`TDLPP_SPPR_public.zip`](TDLPP_SPPR_public.zip), extract it, and open the extracted `TDLPP_SPPR_public` directory in MATLAB. The archive contains the complete source, fixed validation splits/maps, tests and validation records. Raw datasets are external.

```matlab
setup;
report = run_tests(false);
result = run_demo();
```

Tested environment: MATLAB R2023b on Windows, with Statistics and Machine Learning Toolbox. The optional ERS binary is Windows-specific.

The package documents deterministic truncated identity initialization, the frozen numerical eigenvalue-selection behavior and objective-history timing. Its validation uses additional fixed splits and does not establish exact reproduction of manuscript Table 10.

An authors' project-code license has not yet been assigned. ERS retains its upstream non-commercial terms; the legacy PCA helper permissions still need documentation. See `THIRD_PARTY_NOTICES.md` and `IMPLEMENTATION_NOTES.md` inside the archive.

Full English and Chinese usage instructions are included in the archive.
