# Incremental Training of a Recurrent Neural Network Exploiting a Multi-Scale Dynamic Memory
MSLMN code for IAM-OnDB experiments and incremental training. This codebase implements our recurrent model based on a hierarchical recurrent neural network architecture. The model is trained incrementally by dynamically expanding the architecture to capture longer dependencies during training. Each new module is pretrained to maximize its memory capacity.

## References
This work is based on our paper published @ ECML 2020: [https://arxiv.org/abs/2006.16800](https://arxiv.org/abs/2006.16800)

If you find this useful consider citing:
```
@inproceedings{carta2020incremental,
  title={Incremental Training of a Recurrent Neural Network Exploiting a Multi-Scale Dynamic Memory},
  author={Antonio Carta and Alessandro Sperduti and Davide Bacciu},
  booktitle={ECML/PKDD},
  year={2020}
}
```
