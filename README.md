# Active Client Selection for Communication-efficient Federated Learning
Active Client Selection algorithms of Federated Learning implementations by PyTorch in my Thesis.

## Requirements
```shell
torch=1.8.0
torchvision
numpy
scipy
tqdm
h5py
```

## Client Selection methods
```shell
python main.py --method {client selection method you want}
```

 1. ```Random```: Random Selection
 2. ```AFL```: Active Federated Learning [[Jack Goetz et al., 2019](https://arxiv.org/pdf/1909.12641.pdf)]
 3. ```Pow-d```: Power-of-d-Choice [[Yae Jee Cho et al., 2022](https://arxiv.org/pdf/2010.01243.pdf)]
 4. ```Cluster1```: Clustered Sampling 1 [[Yann Fraboni et al., 2021](http://proceedings.mlr.press/v139/fraboni21a/fraboni21a.pdf)]
 5. ```Cluster2```: Clustered Sampling 2 [[Yann Fraboni et al., 2021](http://proceedings.mlr.press/v139/fraboni21a/fraboni21a.pdf)]
 6. ```DivFL```: Diverse Client Selection for FL [[Ravikumar Balakrishnan et al., 2022](https://openreview.net/pdf?id=nwKXyFvaUm)]

## Benchmark Datasets

1. FederatedEMNIST (default)

   Download from this [[link](https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/FederatedEMNIST/download_federatedEMNIST.sh)] and place them in your data directory ```data_dir```.
    
    ```shell
    python src/main.py --dataset FederatedEMNIST --model CNN -A 10 -K 3400 --lr_local 0.1 -B 20 -R 2000 
   ```

2. CelebA
   
   Download from the original CelebA homepage [[link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)].

   ```shell
   python src/main.py --dataset CelebA --model CNN -A 10 -K 9343 --lr_local 0.005 -B 5 -R 100
   ```

3. FederatedCIFAR100
   
   Download from this [[link](https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/fed_cifar100/download_fedcifar100.sh)].

   ```shell
    python src/main.py --dataset FedCIFAR100 --model ResNet -A 10 -K 500 --lr_local 0.1 -B 20 -R 4000 
   ```

4. FederatedCIFAR10 (Partitioned by Dirichlet distribution, followed by Clustered Sampling)

   Don't need to download any dataset.
    
   ```shell
    python src/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 
   ```

5. Reddit

   ```shell
    python src/main.py --dataset Reddit --model BLSTM -A 200 -K 7527 --maxlen 400 \
        --alpha1 0.75 --alpha2 0.01 --alpha3 0.1 \
        --lr_local 0.01 --lr_global 0.001 -E 2 -B 128 -R 100
   ```
   
   - [ ] TODO: reproducible performance

## References
 - https://github.com/FedML-AI/FedML 
 - https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
