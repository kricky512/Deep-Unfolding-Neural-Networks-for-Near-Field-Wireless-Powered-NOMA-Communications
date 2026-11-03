# Running the Near-Field NOMA Simulation

This guide explains how to train the deep unfolding network, compare it with the conventional AO-FP algorithm, and view the simulation results.

## File roles

| File | Role in the workflow |
|---|---|
| `config.py` | Defines the system parameters, training settings, convergence settings, and output directories. |
| `system.py` | Provides channel generation and the physical calculations shared by both methods. |
| `ao_fp.py` | Runs the conventional AO-FP baseline during evaluation. |
| `unfolding.py` | Defines the network used during training and inference. |
| `train.py` | Prepares the datasets, trains the network, and saves the best model. |
| `compare.py` | Evaluates both methods on the same test channels and generates comparison figures. |

The two execution entry points are `train.py` and `compare.py`. The other four files are imported automatically. Run training first so that evaluation can load the saved test dataset and model.

## 1. Prepare the session

Open the project root folder in VS Code. Select and activate the Python environment containing PyTorch and Matplotlib. Run all commands below from the project root.

Review the settings in `config.py` before starting an experiment. The supplied settings are development examples and can be adjusted for the intended simulation scenario.

## 2. Check the workflow with a small run

First, run `python train.py --smoke --check`.

This checks the configuration and performs a network forward pass. It does not train the model or create the full datasets. A successful check reports `forward_check_passed`.

Next, run `python train.py --smoke --device cpu`.

The training process generates independent training, validation, and test datasets, trains the network, and saves the model with the highest validation throughput. Wait for the `Saved best model:` message and for the terminal prompt to return.

Then run `python compare.py --smoke --device cpu --repeats 3`.

The evaluation process loads the saved test dataset and model. For each test channel, it gives both methods the same initial beamformers, evaluates their throughput and execution time, and saves the results and figures.

Use `--smoke` for both training and evaluation. This option selects a separate small scenario with fewer samples, layers, training epochs, and AO iterations. It also adjusts the propagation distances and noise setting. Its purpose is to check the complete workflow; its short runs are not sufficient for final performance conclusions.

## 3. View the small-run results

The small-run datasets are saved in `results/smoke/data/`, and the best model is saved as `results/smoke/checkpoints/best_model.pt`.

Open `results/smoke/results/` in the VS Code Explorer and double-click a figure to view it.

| Output | What to inspect |
|---|---|
| `performance.png` | Mean sum throughput and mean online execution time for the two methods. |
| `paired_throughput.png` | Throughput comparison for individual test channels. |
| `iterations_and_layers.png` | AO iterations and network layers for the first test channel, shown separately. |
| `training_curve.png` | Training and validation throughput across epochs. |
| `comparison.csv` | Per-sample throughput, execution time, power error, and AO iteration information. |
| `comparison_summary.json` | Aggregate results, evaluation settings, and the AO convergence fraction. |
| `training_history.json` | The recorded training and validation scores. |
| `iteration_traces.json` | The numerical traces used for the iteration and layer figure. |

Higher throughput and lower execution time are preferable. Check `ao_converged_fraction` before interpreting the baseline results; increase the iteration limits if necessary.

## 4. Run the configured experiment

After checking the small run and reviewing `config.py`, run `python train.py --device cuda`, followed by `python compare.py --device cuda --repeats 3`.

Use `--device cpu` instead if CUDA is unavailable. During comparison, both methods use the selected device and the same numerical precision.

Without `--smoke`, datasets are saved in `data/`, the best model in `checkpoints/`, and figures and evaluation records in `results/`. These outputs are separate from the small-run outputs.

## 5. Repeat or change an experiment

To repeat evaluation with the same configuration and saved model, run `compare.py` again using the corresponding options. Retraining is not required.

When an existing dataset matches the configuration, training reuses it. If the system parameters or dataset sizes change, the program rejects incompatible saved data. After confirming the new settings, run `python train.py --device cuda --regenerate-data`, then run the matching evaluation command.

Changing the antenna or user count requires training a network with the corresponding dimensions. Evaluation checks that the test data, model, and current system configuration match.

A new training run updates the best-model file and training history in its output directory. Preserve earlier experiment outputs before reusing that directory if they are needed for comparison. The saved best model may come from epoch zero if training does not improve validation throughput.
