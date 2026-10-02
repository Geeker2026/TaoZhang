# TDLPP + SPPR — MATLAB source package

Download [`TDLPP_SPPR_public.zip`](TDLPP_SPPR_public.zip), extract it, and open the extracted `TDLPP_SPPR_public` directory in MATLAB. The archive contains the complete source, fixed validation splits/maps, tests and validation records. Raw datasets are external.

```matlab
setup;
report = run_tests(false);
result = run_demo();
```

Tested environment: MATLAB R2023b on Windows, with Statistics and Machine Learning Toolbox. The optional ERS binary is Windows-specific.

The package documents deterministic truncated identity initialization, the frozen numerical eigenvalue-selection behavior and objective-history timing. 
