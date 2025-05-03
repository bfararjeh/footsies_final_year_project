from itertools import product
import json, os

# baseline config structure
baseConfig = {
    "experiment_name": "",
    "model": {
        "LSTM_unit_size": None,
        "dense_unit_size": None,
        "dropout_rate": None,
    },
    "training": {
        "batch_size": None,
        "sequence_length": None,
        "step": 1
    },
}

# hyperparams
lstmSizes = [16, 32, 64]
denseSizes = [16, 32]
dropouts = [0.3, 0.5]
batchSizes = [32, 64]
seqLengths = [10, 20, 30]
stepSizes = [1, 2]

outputDirectory = os.path.join(os.path.dirname(__file__), "hyperparameters")

# create config permutations
count = 1
for (seq, steps, lstm, dense, dr, bs) in product(
    seqLengths, stepSizes, lstmSizes, denseSizes, dropouts, batchSizes):

    config = baseConfig.copy()
    config["experiment_name"] = f"exp_{count:03d}"
    
    config["model"]["LSTM_unit_size"] = lstm
    config["model"]["dense_unit_size"] = dense
    config["model"]["dropout_rate"] = dr

    config["training"]["batch_size"] = bs
    config["training"]["sequence_length"] = seq
    config["training"]["step"] = steps

    # save as json
    with open(os.path.join(outputDirectory, f"{config['experiment_name']}.json"), 'w') as f:
        json.dump(config, f, indent=2)

    count += 1

print(f"Generated {count-1} config files in '{outputDirectory}'")