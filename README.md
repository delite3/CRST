# carla-radar-sim #

A simple radar simulator built around the semantic ray cast sensor in Carla.

## Requirements

* Carla 0.9.14
* Python 3.8

This has only been tested on Ubuntu.

## Installation

### Carla

[Download](https://github.com/carla-simulator/carla/releases/tag/0.9.14) and extract Carla 0.9.14 somewhere nice.

### Python (Ubuntu)

If you don't have Python 3.8, install it. This can be done on Ubuntu-based systems using
```sh
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv
```

It's best to create a virtual environment and use it to run/develop the scripts
```sh
python3.8 -m venv venv
source venv/bin/activate
```

### Python (Windows)

Download and install the latest release of Python 3.8 from https://www.python.org/downloads/.

Create a virtual environment and use it to run/develop the scripts
```sh
py -3.8 -m venv venv
./venv/Scripts/Activate.ps1
```

### Requirements

Finally, update `pip` and install the python requirements
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

First, start the Carla server
```sh
/PATH/TO/CARLA/CarlaUE4.sh
```

Start the simulator
```
python radar_sim.py
```
 
To get the CLI-options
```
python radar_sim.py --help
```

There's a lot of hotkeys in the `Radar` window (the point cloud view). The custom hotkeys are printed at startup, Open3Ds built-in hotkeys can be printed by pressing `h`.


## Typing
Carla Python stubs are included in `./typings/carla`. They were downloaded from https://github.com/aasewold/carla-python-stubs.