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

  ```bash
  python3 bioINet.py --dataset <dataset name>
  ```
  - Examples
    ```bash
    python3 main.py --dataset mnist
    python3 main.py --dataset fmnist
    python3 main.py --dataset cifar10
    python3 main.py --dataset cifar100
    python3 main.py --dataset permuted_mnist
    python3 main.py --dataset rotated_mnist
    ```
 
## To run the BioINet algorithm from dataset folder

- Move to the desired folder (e.g cd MNIST_Exp)

```bash
python3 main.py --dataset mnist
```

## To run the benchmark algorithms (EWC, LWF, SI, Naive, joint) from dataset folder

- Move to the desired folder (e.g cd MNIST_Exp/Benchmark_modelExp)

  ```bash
  python3 benchmarkSeparateWithGeneratorForBuffer.py
  ```
  
## To cite the paper
  ```bash
  ```
