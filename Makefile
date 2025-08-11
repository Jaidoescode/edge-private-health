PY=python

install:
	pip install -r requirements.txt

prep:
	$(PY) -m src.data.prepare --root ./data --seed 42

train:
	$(PY) -m src.train.train_baseline --data_root ./data --epochs 3

fed:
	$(PY) -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

eval:
	$(PY) -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
