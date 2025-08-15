from pytorch_lightning.utilities import rank_zero_only
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from pytorch_lightning.cli import LightningCLI
from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
from models import BaseModel
import os
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score


import torchmetrics.functional as MF
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for predicting.")
        parser.add_argument("--do_train", type=bool,
                            help="If True: skip training the model.")
        parser.add_argument("--do_predict", type=bool,
                            help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                            default=False, help="If True: compute val metrics.")

    def before_instantiate_classes(self):
        # The number of features is only known inside the data module, but we need that info to instantiate the model.
        # Since datamodule and model are instantiated at the same time with LightningCLI, we need to set the number of features here.
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features)
        self.config.model.init_args.n_channels = n_features

        # The exact positive class weight changes with the data fold in the data module, but the weight is needed to instantiate the model.
        # Non-fire pixels are marked as missing values in the active fire feature, so we simply use that to compute the positive class weight.
        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)

        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        return

    def before_test(self):
        return

    def before_validate(self):
        return 


def brier_score(preds, target, class_wise=False):
    """
    Computes the Brier score for binary classification, with an option
    for class-wise scoring.

    Args:
        y_hat (np.ndarray or torch.Tensor): Predicted probabilities
                                            (e.g., from a sigmoid output),
                                            should be between 0 and 1.
        y_true (np.ndarray or torch.Tensor): True labels (0 or 1).
        class_wise (bool): If True, returns a tuple with the Brier scores
                           for the negative (0) and positive (1) classes.
                           Defaults to False.

    Returns:
        float or tuple: The total Brier score, or a tuple of (score_neg, score_pos).
    """
    y_hat = torch.sigmoid(preds).flatten()
    y_true = target.flatten()

    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_hat = np.asarray(y_hat)
    y_true = np.asarray(y_true)

    # Check for valid input values
    if not np.all((y_hat >= 0) & (y_hat <= 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("True labels must be 0 or 1.")

    if not class_wise:
        return np.mean((y_hat - y_true)**2)
    else:
        # Separate scores for each class
        neg_mask = y_true == 0
        pos_mask = y_true == 1

        neg_score = np.mean((y_hat[neg_mask] - 0)**2) if np.any(neg_mask) else 0.0
        pos_score = np.mean((y_hat[pos_mask] - 1)**2) if np.any(pos_mask) else 0.0
        
        return neg_score, pos_score


def plot_calibration(preds: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    """
    Produces a combined confidence calibration plot for binary classification.

    Args:
        y_hat (torch.Tensor): A tensor of predicted probabilities (from a sigmoid output).
        y (torch.Tensor): A tensor of true binary labels (0s and 1s).
        n_bins (int): The number of bins to use for the plot.
    """
    color_count = '#d64161'
    color_tpr = 'teal'
    y_hat = torch.sigmoid(preds).flatten()
    y = target.flatten()
    
    # 2. Convert to numpy for easier array operations
    y_hat_np = y_hat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # 3. Get predicted class and confidence
    y_pred_np = (y_hat_np > 0.5).astype(int)
    
    # 4. Bin the predictions based on confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_hat_np, bins) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1 # Handle max confidence edge case
    
    # 5. Calculate metrics for each bin
    bin_acc = np.zeros(n_bins)
    bin_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        
        if np.any(bin_mask):
            bin_counts[i] = np.sum(bin_mask)
            bin_acc[i] = np.mean(y_np[bin_mask]) # Accuracy is % of true positives in the bin
            bin_prob[i] = np.mean(y_hat_np[bin_mask])


    # 6. Calculate ECE
    total_samples = len(y_np)
    if total_samples == 0:
      ece_val = 0
      uce_val = 0
    else:
      ece_val = np.sum(bin_counts / total_samples * np.abs(bin_acc - bin_prob))
      uce_val = np.mean(np.abs(bin_acc - bin_prob))

    # 7. Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Calibration Plot (Accuracy vs. Confidence)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax1.plot(bin_prob, bin_acc, marker='s', label='Model Calibration', color=color_tpr)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('True Positive Rate', color= color_tpr)
    #ax1.set_xlim(0, 1)
    #ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.tick_params(axis='y', labelcolor=color_tpr)
    
    print(bins)
    print(bin_counts)
    # Create a second y-axis for the histogram
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1], bin_counts, width=1/n_bins, align='edge', edgecolor='black',color=color_count, alpha=0.3, label='Sample Count')
    ax2.set_ylabel('Count (pixels)', color=color_count)
    ax2.tick_params(axis='y', labelcolor=color_count)
    ax2.set_yscale('log')

    # Set the title with the ECE value
    fig.suptitle(f'Probability Calibration Plot | ECE: {ece_val:.4f} | UCE: {uce_val:.4f}')
    
    # Manually create a legend combining both plots
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='upper left')

    plt.tight_layout()
    return fig

def main():

    # LightningCLI automatically creates an argparse parser with required arguments and types,
    # and instantiates the model and datamodule. For this, it's important to import the model and datamodule classes above.
    cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, 
                         parser_kwargs={"parser_mode": "yaml"}, run=False,
                         save_config_callback=None
                         )

    if cli.config.do_train:
        pass       
  
    if cli.config.do_validate:
        pass

    if cli.config.do_test:
        pass

    # Load trained weights 
    ckpt = cli.config.ckpt_path

    # Produce predictions, save them in a single file, including ground truth fire targets and input fire masks.
    prediction_output = cli.trainer.predict(
        cli.model, cli.datamodule, ckpt_path=ckpt)
    #x_af = torch.cat(
    #    list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
    y = torch.cat(list(map(lambda tup: tup[1].flatten(), prediction_output)), axis=0)
    y_hat = torch.cat(
        list(map(lambda tup: tup[2].flatten(), prediction_output)), axis=0)
    
    print("y_hat shape:", y_hat.shape)
    print("y shape:", y.shape)
    
    print("Computing metrics...")
    ap = MF.average_precision(y_hat, y, task="binary")
    """
    ece = MF.classification.binary_calibration_error(
        y_hat, y, 
    )
    ece_class0 = MF.classification.binary_calibration_error(
        y_hat, y, ignore_index=1
    )
    ece_class1 = MF.classification.binary_calibration_error(
        y_hat, y, ignore_index=0
    )
    sce = 0.5 * (ece_class0 + ece_class1)
    """
    
    b = brier_score(y_hat, y, class_wise=False)
    b0, b1 = brier_score(y_hat, y, class_wise=True)
    acc = MF.classification.binary_accuracy(y_hat, y)

    pt, pp = calibration_curve(
        y.detach().flatten().cpu().numpy(), torch.sigmoid(y_hat).detach().flatten().cpu().numpy(), n_bins=15, strategy='uniform'
    )
    ap_sk = average_precision_score(y.detach().flatten().cpu().numpy(), torch.sigmoid(y_hat).detach().flatten().cpu().numpy())

   # p, r, t = precision_recall_curve(
   #     y.detach().flatten().cpu().numpy(), torch.sigmoid(y_hat).detach().flatten().cpu().numpy()
   # )
    f1 = f1_score(
        y.detach().flatten().cpu().numpy(), (torch.sigmoid(y_hat).detach().flatten().cpu().numpy() > 0.5).astype(int)
    )
    #print("Thresholds SKL", t)
    print("F1 Score:", f1)
    print("Average Precision:", ap_sk)
    print("sk UCE:" , np.mean(np.abs(pt - pp)))

    print("accuracy:", acc.item())
    print("Average Precision:", ap.item())
    #print("ECE:", ece.item())
    #print("SCE:", sce.item())
    #print("ECE class 0:", ece_class0.item())
    #print("ECE class 1:", ece_class1.item())
    print("Brier Score:", b)
    print("Brier Score class 0:", b0)
    print("Brier Score class 1:", b1)
    fig = plot_calibration(y_hat, y)
    fig.savefig('test.pdf', bbox_inches='tight')

    
    #fire_masks_combined = torch.cat(
        #$[x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

    #a = 1
    #predictions_file_name = os.path.join(
    #    cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
    #torch.save(fire_masks_combined, predictions_file_name)


if __name__ == "__main__":
    main()
