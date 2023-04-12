from argparse import ArgumentParser
import MNIST_Exp.mainDemoRun as mnist_run
import FMNIST_Exp.mainDemoRun as fmnist_run
import CIFAR10_Exp.mainDemoRun as cifar10_run
import CIFAR100_Exp.mainDemoRun as cifar100_run
import Permuted_MNIST_Exp.mainDemoRun as pmnist_run
import Rotated_MNIST_Exp.mainDemoRun as rmnist_run

def main():
    arg_choices = ['mnist','fmnist','cifar10','cifar100','permuted_mnist','rotated_mnist']
    parser = ArgumentParser(description="BioINet Algorithm")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to perform experiment on",choices=arg_choices)
    args = parser.parse_args()

    if args.dataset=="mnist":
        mnist_run.main()
    elif args.dataset=="fmnist":
        fmnist_run.main()
    elif args.dataset=="cifar10":
        cifar10_run.main()
    elif args.dataset=="cifar100":
        cifar100_run.main()
    elif args.dataset=="permuted_mnist":
        pmnist_run.main()
    elif args.dataset=="rotated_mnist":
        rmnist_run.main()
    else:
        print("You have passed an unsupported dataset parameter")

if __name__=="__main__":
    main()
