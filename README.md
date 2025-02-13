This is an implementation for the PIecewise Linear Organic Tree (PILOT), a linear model tree algorithm proposed in the paper Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. Machine Learning, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

This repository also includes the implementation for RaFFLE, a random forest of PILOT trees:
Raymaekers, J., Rousseeuw, P. J., Servotte, T., Verdonck, T., & Yao, R. (2025). A Powerful Random Forest Featuring Linear Extensions (RaFFLE). _Under Review_

### Requirements:
Requirements can be installed by running
```
pip install -r requirements.txt
```
The RaFFLE implementation uses a c++ version of pilot for computational speed.
To build the c++ wrapper, follow these steps:

1. Make sure the necessary dependencies are installed
    ```
    sudo apt-get update
    sudo apt-get install cmake g++ libopenblas-dev liblapack-dev
    ```

2. Install Armadillo
    ```
    wget http://sourceforge.net/projects/arma/files/armadillo-14.0.3.tar.xz
    tar -xvf armadillo-14.0.3.tar.xz
    cd armadillo-14.0.3/
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    ```

3. Install pybind and add to cmake config
    ```
    pip install pybind11
    ``` 
    Locate the cmake config: 
    ```
    pip show pybind11
    ```
    copy the installation path to `CMakeLists.txt`: 
    ```
    set(pybind11_DIR <path>/pybind11/share/cmake/pybind11)
    ```

4. Install carma
    Clone the repo: `git@github.com:RUrlus/carma.git`
    Build the package:
    ```
    cd carma
    mkdir build
    cd build
    cmake -DCARMA_INSTALL_LIB=ON ..
    cmake --build . --config Release --target install
    ```

5. Build the wrapper
    ```
    cd PILOT/build
    cmake ..
    make
    ```

    If you get errors like this: 
    ``` 
    Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS NumPy)
    ```
    You might have to add these lines to `CMakeLists.txt`:
    ```
    set(Python3_INCLUDE_DIR <path_to_python_include>)
    set(Python3_NumPy_INCLUDE_DIR <pat_to_numpy_include>)
    ```

    The first can be found by running
    ```
    python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])"
    ```
    The latter can be found by running 
    ```
    python3 -c "import numpy; print(numpy.get_include())"
    ```

    The outputs of these commands should replace `<path_to_python_include>` and `<path_to_numpy_include>` respectively.


### Example
You can run an example for RaFFLE with the [raffle_example.py](raffle_example.py) script.

### RaFFLE benchmark
To run the same benchmark as described in the RaFFLE paper, you first need to download all the benchmark datasets using the [download_data.py](download_data.py) script.

```
python download_data.py
```

Next you can run benchmark by running the [benchmark.py](benchmark.py) script:
```
python benchmark.py
```

Results will be stored in the `Output` folder.

The plots from the paper are created with the [paperplots.py](paperplots.py) script. You can create all plots by running:

```
python paperplots.py --all
```

