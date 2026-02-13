# 1D Burgers Riemann Solver (PyClaw + Panel)

Interactive visualization of the 1D inviscid Burgers equation Riemann problem using:

* PyClaw (finite volume solver)
* Panel (interactive dashboard)
* Matplotlib (plotting)
* Runs inside **WSL Ubuntu**
* Viewed from a browser on Windows host

PyClaw docs:
https://www.clawpack.org/pyclaw/tutorial.html
https://www.clawpack.org/pyclaw/rp.html#module-clawpack.riemann.burgers_1D_py

---

# Overview

This project solves the 1D inviscid Burgers equation:

$$
u_t + \left(\frac{u^2}{2}\right)_x = 0
$$

with Riemann initial data:

$$
u(x,0) =
\begin{cases}
u_L & x < 0 \
u_R & x > 0
\end{cases}
$$

The app allows:

* Switching between rarefaction (u_L < u_R)
* Shock (u_L > u_R)
* Interactive time evolution
* Numerical vs exact solution comparison

---

# System Requirements

* Windows 10 or 11
* WSL2
* Ubuntu (22.04 recommended)
* Python ≥ 3.10

---

# Step 1 - Install WSL (If Not Already Installed)

In Windows PowerShell (Admin):

```powershell
wsl --install
```

Reboot if prompted.

Install Ubuntu from Microsoft Store.

---

# Step 2 - Create Python Virtual Environment

Inside WSL:

```bash
mkdir -p ~/burgers-panel
cd ~/burgers-panel

python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# Step 3 - Install Dependencies

Install required packages:

```bash
pip install numpy matplotlib panel clawpack watchfiles
```

Explanation:

* `clawpack` → includes PyClaw
* `panel` → dashboard framework
* `watchfiles` → enables fast autoreload
* `matplotlib` → plotting
* `numpy` → numerical arrays

---

# Step 4 - Run the Panel App

From project directory:

```bash
panel serve app.py --autoreload --address 0.0.0.0 --port 5006
```

Expected output:

```
Bokeh app running at:
http://0.0.0.0:5006/app
```

---

# Step 5 - Open in Windows Browser

In Windows (Chrome / Firefox / Edge), open:

```
http://localhost:5006/app
```

WSL automatically forwards the port to Windows.

We should see the interactive Burgers solver.

---

# How It Works

```
Windows Browser
        ↓
localhost:5006
        ↓
WSL Ubuntu
        ↓
Panel (Bokeh Server)
        ↓
PyClaw Solver
```

* Panel runs a Bokeh web server.
* PyClaw computes the numerical solution.
* Matplotlib renders plots using the Agg backend.
* Windows browser connects via localhost.

---

# Common Issues

## Matplotlib Backend Error

Error:

```
The Matplotlib backend is not configured
```

Solution:
Ensure this appears before importing pyplot:

```python
import matplotlib
matplotlib.use("Agg")
```

---

## Port Already in Use

Kill old server:

```bash
lsof -i :5006
kill -9 <PID>
```

Or run on another port:

```bash
panel serve app.py --port 5010
```

Then visit:

```
http://localhost:5010/app
```

---

## Clawpack Installation Issues

If installation fails:

```bash
pip install cython
pip install clawpack
```

If still failing, ensure:

```bash
sudo apt install build-essential gfortran
```
