This is an implementation for the PIecewise Linear Organic Tree (PILOT), a linear model tree algorithm proposed in the paper Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by PILOT. Machine Learning, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

### Requirements:
numba==0.55.1  
numpy==1.21.2  
pandas==1.4.1


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




