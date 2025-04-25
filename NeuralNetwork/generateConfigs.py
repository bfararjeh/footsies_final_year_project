from itertools import product
import json, os

# baseline config structure
baseConfig = {
    "experiment_name": "",
    "model": {
        "LSTM_unit_size": None,
        "dense_unit_size": None,
        "dropout_rate": None,
        "learning_rate": None,
        "early_stopping_patience": 3
    },
    "training": {
        "batch_size": None,
        "epochs": 15,
        "sequence_length": None,
        "step": 1
    },
    "feature_engineering": {
        "enable_isDead": None,
        "enable_velocity": None,
        "enable_relativeDistance": None
    }
}

# hyperparams
lstmSizes = [32, 64]
denseSizes = [16, 32]
dropouts = [0.1, 0.2, 0.5]
learningRates = [0.001, 0.0005]
batchSizes = [32, 64]
seqLengths = [20, 50]
stepSizes = [1, 5]

# feature toggles
featureOptions = [
    {"enable_isDead": 1, "enable_velocity": 1, "enable_relativeDistance": 0},
    {"enable_isDead": 1, "enable_velocity": 1, "enable_relativeDistance": 1},
    {"enable_isDead": 0, "enable_velocity": 0, "enable_relativeDistance": 1}
]

outputDirectory = os.path.join(os.path.dirname(__file__), "experimentConfigs")

# create config permutations
count = 1
for (lstm, dense, dr, lr, bs, seq, steps, feats) in product(
    lstmSizes, denseSizes, dropouts, learningRates, batchSizes, seqLengths, stepSizes, featureOptions):

    config = baseConfig.copy()
    config["experiment_name"] = f"exp_{count:03d}"
    
    config["model"]["LSTM_unit_size"] = lstm
    config["model"]["dense_unit_size"] = dense
    config["model"]["dropout_rate"] = dr
    config["model"]["learning_rate"] = lr

    config["training"]["batch_size"] = bs
    config["training"]["sequence_length"] = seq

    config["feature_engineering"] = feats.copy()

    # Save as JSON
    with open(os.path.join(outputDirectory, f"{config['experiment_name']}.json"), 'w') as f:
        json.dump(config, f, indent=2)

    count += 1

print(f"Generated {count-1} config files in '{outputDirectory}'")