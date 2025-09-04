1. general flow 

- screenshot
- data labelling -> roboflow (download ratio train/valid/test: 7:2:1)
- download dataset
- use yolo script to train
- test_model, can choose data not in dataset uploaded before
- write auto script

2. Q&A
torch does not recognize cuda, need to use cu128(least or nightly) version torch

3. UV env
