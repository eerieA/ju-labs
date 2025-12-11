# Dependencies

Julia 1.12.2

Julia packages
- BenchmarkTools
- Plots
- ProgressMeter

# How to set env var for number of threads

Linux/IOS

```bash
export JULIA_NUM_THREADS=8
```

Windows Power shell

```bash
$env:JULIA_NUM_THREADS="8"
```

Set these before running `jupyter lab`.

There are several ways to tell Julia about number of threads when working with Jupyter Lab. This is just the simplest one of them.
