import os
import re
import matplotlib.pyplot as plt

# Function to read the log file and extract losses
def extract_losses(log_file):
    epoch_pattern = re.compile(r"INFO:classification_training.fine_tuning_vit:Epoch \[(\d+)/\d+\]")
    batch_loss_pattern = re.compile(r"INFO:classification_training.fine_tuning_vit:\s+Batch \[(\d+)/\d+\], Loss: ([\d.]+), Time: [\d.]+")
    val_loss_pattern = re.compile(r"INFO:classification_training.fine_tuning_vit:Validation Loss: ([\d.]+), Validation Accuracy: ([\d.]+)%")

    epochs = []
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    current_epoch = None
    last_batch_loss = None

    with open(log_file, 'r') as file:
        for line in file:
            epoch_match = epoch_pattern.match(line)
            if epoch_match:
                if current_epoch is not None and last_batch_loss is not None:
                    training_losses.append(last_batch_loss)
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                last_batch_loss = None

            batch_loss_match = batch_loss_pattern.match(line)
            if batch_loss_match:
                last_batch_loss = float(batch_loss_match.group(2))

            val_loss_match = val_loss_pattern.match(line)
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
                val_acc = float(val_loss_match.group(2))
                validation_losses.append(val_loss)
                validation_accuracies.append(val_acc)

    if current_epoch is not None and last_batch_loss is not None:
        training_losses.append(last_batch_loss)

    return epochs, training_losses, validation_losses, validation_accuracies

# Function to calculate average loss per epoch
def calculate_average_losses(epoch_losses):
    avg_losses = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_losses.items()}
    return avg_losses



# Function to plot Loss vs Epoch and Accuracy vs Epoch
def plot_metrics(epochs, training_losses, validation_losses, validation_accuracies, outputdir):
    # Plot Training Loss vs. Epoch
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss', linestyle='--', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot Validation Accuracy vs. Epoch
    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='green', linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outputdir,"train.png"))
    plt.show()

# Path to your log file
log_file_path = '/home/jcuomo/CameraTraps/output_step3.txt'
output_dir = "/home/jcuomo/CameraTraps/output/classification/step3"

# Extract losses and accuracies from the log file
epochs, training_losses, validation_losses, validation_accuracies = extract_losses(log_file_path)

# Plot the extracted metrics
plot_metrics(epochs, training_losses, validation_losses, validation_accuracies, output_dir)
print(f"Max accuracy={max(validation_accuracies)} at epoch {epochs[validation_accuracies.index(max(validation_accuracies)) + 1]}")