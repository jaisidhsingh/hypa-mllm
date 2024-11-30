# Training MLLM with Hyper-Alignment

## Notes

### Single GPU trials:

Llama-3.2-1B and Qwen-2.5-1.5B loaded together take 14GB of VRAM.

Connector training takes 6 hours for LLama-3.2-1B (max `batch_size=32` on 44GB VRAM).
Connector+LLM training takes 9 hours for LLama-3.2-1B (max `batch_size=16` on 44GB VRAM). 

## Todos

- [x] Set up repo
- [x] Set up MLLM modelling
- [x] Write `src/model/model_tests.py`
- [x] Test `src/model/model_tests.py`
- [x] Test `data/alignment_datasets.py`
- [x] Write `src/training/trainers.py` -tmp
- [x] Test `src/training/trainers.py` -tmp
- [x] Write `train_mllm.py` -tmp
- [x] Test `train_mllm.py` -tmp
- [ ] Write/see a full trainer with eval+schedules
- [ ] Write `src/model/hypermllm.py`
- [ ] Write `train_hypermllm.py` 
