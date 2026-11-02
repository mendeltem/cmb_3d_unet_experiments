# The easiest way: four short Python scripts

Run them from the repository folder, in this order (after `./install.sh` and `source env.sh`):

```bash
python examples/easy/1_prepare_data.py     # five VALDO example scans -> train (3) and test (2)
python examples/easy/2_train.py            # train a model from scratch
python examples/easy/3_test.py             # test the pretrained model (or: 3_test.py <model folder>)
python examples/easy/4_fine_tune.py        # fine-tune the pretrained model and test it
```

Or all of them in one go: `python examples/easy/run_all.py` (stops at the first failing step; finished steps are reused).

Each script is a few lines that call `cmbx.easy`:

| function | what it does |
|---|---|
| `easy.preprocess(scans, out, ...)` | raw scans -> one folder per case |
| `easy.split(data, out, train=[...], test=[...])` | copy cases into train / test |
| `easy.train(data, model, epochs=60)` | train a new model |
| `easy.fine_tune(data, model, start_from="pre", epochs=30)` | adapt the pretrained (or your) model |
| `easy.predict(data, model, out)` | find microbleeds; `model="pre"` = the pretrained model |
| `easy.score(predictions, data)` | found / false alarms / missed / F1 per case and in total |
| `easy.cross_validate(data, out, folds=5)` | k-fold cross-validation |

Every function runs the matching `cmbx` command, so any command-line option works as a keyword:
`easy.train(data, model, csf_rule=True, device="cpu")` (all options: `python -m cmbx train --help`).
The step-by-step versions that show what happens inside are in `examples/python/`.
