# Logiformer

This is a model to tackle the logical reasoning task in the field of multiple-choice machine reading comprehension.

**Logiformer: A Two-branch Graph Transformer Network for Interpretable Logical Reasoning** (*SIGIR 2022*) [[paper]](https://arxiv.org/abs/2205.00731)

## How to Run

```
sh run_logiformer.sh
```

## Experiment Results

It achieves the **state-of-the-art (SOTA)** results compared with all the RoBERTa-Large single model methods. Also, it ranks on the **9th place** in the leaderboard compared with other larger models.

## Acknowledgement

The implementation of Logiformer is inspired by [DAGN](https://arxiv.org/abs/2103.14349) and [LReasoner](https://arxiv.org/abs/2105.03659) and supported by [Huggingface Toolkit](https://huggingface.co/docs/transformers/index).

## Citation

```
@article{xu2022logiformer,
  title   = {Logiformer: A Two-Branch Graph Transformer Network for Interpretable Logical Reasoning},
  author  = {Xu, Fangzhi and Liu, Jun and Lin, Qika and Pan, Yudai and Zhang, Lingling},
  journal = {arXiv preprint arXiv:2205.00731},
  year    = {2022}
}
```
