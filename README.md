## VSEPR Code Repository
Code repository for my research project, Developing a Computer Simulation For Estimating Lowest-Energy VSEPR Geometries Using the Thomson Problem.

### Running the Program
Program is only tested on Linux.<br/>
First clone the repository:
```sh
git clone https://github.com/nijeff27/VSEPR.git
cd VSEPR
```
It is suggested to create a virtual environment via `venv` (`conda` can also work):
```sh
python -m venv .venv
source .venv/bin/activate
```
Then install the required packages:
```sh
pip install -r requirements.txt
```
On Linux machines using the Wayland protocol (if not sure, run `echo "$XDG_SESSION_TYPE"`), make sure to run the program as follows (as Open3D only supports the X11 protocol as of 3/12/2026):
```
XDG_SESSION_TYPE=x11 python vsepr.py
```
The program should work on other operating systems.

### Credits
* `numpy` ([https://numpy.org/](https://numpy.org/))
* `open3d` ([https://www.open3d.org/](https://www.open3d.org/))
* `openpyxl` ([https://openpyxl.readthedocs.io/en/stable/index.html](https://openpyxl.readthedocs.io/en/stable/index.html))