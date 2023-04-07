# BioINet
Implementation of the Biological Inspired Network (BioINet) for continual learning applications. More details about the algorithm and the architecture is provided in the paper "".

## Dataset Link

- [MNIST]()
- [FMNIST]()
- [CIFAR10]()
- [CIFAR100]()
- [PMNIST]()
- [RMNIST]()

## Install the dependencies in a virtual environment

- Create a virtual environment (Python version 3.8.10) 
  
  ```bash
  python3 -m venv BioINet
  ```

- Activate the virtual environment
  ```bash
  . BioINet/bin/activate
  
- Install the dependencies

  ```bash
  pip3 install -r requirements.txt
  ```
 
## To run the BioINet algorithm

- Move to the desired folder (e.g cd MNIST_Exp)

```bash
python3 main.py
```

## To run the benchmark algorithms (e.g cd MNIST_Exp/Benchmark_modelExp)
  
  ```bash
  python3 benchmarkSeparateWithGeneratorForBuffer.py
  ```
  
## To cite the paper
  ```bash
  ```
