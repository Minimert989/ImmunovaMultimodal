import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Assuming these are in the same directory or accessible via PYTHONPATH
from dataset import PatientDataset
from model import ImmunovaMultimodalModel
from Immunova2_Module1.model import TILBinaryCNN
from pycox.models import DeepSurv
from sklearn.model_selection import train_test_split

def call_model(model_type: str, **kwargs):
    """
    A modular function to call different types of models for inference.

    Args:
        model_type (str): The type of model to call (e.g., 'multimodal', 'til_cnn', 'deepsurv').
        **kwargs: Additional arguments specific to the model type.

    Returns:
        dict: A dictionary containing prediction results.
    """
    if model_type == 'multimodal':
        return _call_multimodal_model(**kwargs)
    elif model_type == 'til_cnn':
        return _call_til_cnn_model(**kwargs)
    elif model_type == 'deepsurv':
        return _call_deepsurv_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def _call_deepsurv_model(
    model_path: str = "Immunova2_module2/trained_deepsurv_model.pt",
    data_path: str = "Immunova2_module2/ALL_TCGA.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Calls the DeepSurv model for prediction.

    Args:
        model_path (str): Path to the trained DeepSurv model weights.
        data_path (str): Path to the CSV file containing the full dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for train-test split.

    Returns:
        pd.DataFrame: DataFrame containing survival predictions.
    """
    import pandas as pd
    import numpy as np
    import torch
    import torchtuples as tt
    from pycox.models import DeepSurv
    from sklearn.model_selection import train_test_split

    # Load processed data
    full_data = pd.read_csv(data_path)
    durations = full_data["overall_survival"]
    events = full_data["status"]
    X = full_data.drop(columns=["case_id", "overall_survival", "status"])

    # Convert -1.0 to NaN and drop
    valid_mask = (durations != -1.0)
    X = X.loc[valid_mask]
    durations = durations.loc[valid_mask]
    events = events.loc[valid_mask]

    # Split data
    X_train, X_test, durations_train, durations_test, events_train, events_test = train_test_split(
        X, durations, events, test_size=test_size, random_state=random_state
    )

    # Log-transform durations
    durations_train = np.log1p(durations_train)
    durations_test = np.log1p(durations_test)

    # Load trained DeepSurv model
    model = DeepSurv.load_net(model_path)

    # Predict survival
    surv = model.predict_surv_df(X_test)

    return surv

def _call_til_cnn_model(
    model_path: str = "Immunova2_Module1/til_model.pth",
    h5_file_path: str = "Immunova2_Module1/acc-h5/combined.h5",
    batch_size: int = 16
):
    """
    Calls the TILBinaryCNN model for prediction.

    Args:
        model_path (str): Path to the trained TIL model weights.
        h5_file_path (str): Path to the H5 file containing images and labels.
        batch_size (int): Batch size for DataLoader.

    Returns:
        pd.DataFrame: DataFrame containing 'true_label', 'predicted_label', and 'probability' predictions.
    """
    import torch
    import h5py
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from Immunova2_Module1.model import TILBinaryCNN # Assuming model.py is in Immunova2_Module1

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = TILBinaryCNN(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with h5py.File(h5_file_path, "r") as f:
        images = torch.tensor(f["images"][:], dtype=torch.float32)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)

    test_dataset = TensorDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            preds_np = preds
            if preds_np.ndim == 0:
                preds_np = np.expand_dims(preds_np, axis=0)
            y_pred.extend(preds_np)
            y_true.extend(y.numpy())
            probs_np = probs.cpu().numpy()
            if probs_np.ndim == 0:
                probs_np = np.expand_dims(probs_np, axis=0)
            y_scores.extend(probs_np)

    return pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_scores
    })

def _call_multimodal_model(
    model_path: str = "model.pth",
    wsi_feature_path: str = "wsi_feature/wsi_feature.pkl",
    omics_feature_path: str = "omics_feature/omics_dict.pkl",
    label_feature_path: str = "label_feature/label_dict_ACC.pkl",
    val_ids_path: str = "val_ids.txt",
    input_dims: dict = None,
    task: str = "response",
    batch_size: int = 32
):
    """
    Calls the ImmunovaMultimodalModel for prediction.

    Args:
        model_path (str): Path to the trained model weights.
        wsi_feature_path (str): Path to the WSI features pickle file.
        omics_feature_path (str): Path to the omics features pickle file.
        label_feature_path (str): Path to the label features pickle file.
        val_ids_path (str): Path to the validation IDs text file.
        input_dims (dict): Dictionary specifying input dimensions for omics data.
        task (str): The prediction task ('response', 'til', or 'survival').
        batch_size (int): Batch size for DataLoader.

    Returns:
        pd.DataFrame: DataFrame containing 'id', 'label', and 'prob' predictions.
    """
    if input_dims is None:
        input_dims = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with open(wsi_feature_path, "rb") as f:
        wsi_features = pickle.load(f)

    with open(omics_feature_path, "rb") as f:
        omics_dict = pickle.load(f)

    with open(label_feature_path, "rb") as f:
        label_dict = pickle.load(f)

    with open(val_ids_path, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]

    val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, input_dims)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = ImmunovaMultimodalModel(input_dims=input_dims)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["id"]
            wsi_feat = batch["wsi_feat"]
            if wsi_feat is not None:
                wsi_feat = wsi_feat.to(device)

            rna = batch["rna"].to(device)
            methyl = batch["methyl"].to(device)
            prot = batch["protein"].to(device)
            mirna = batch["mirna"].to(device)

            response_label = batch["response_label"].to(device)
            til_label = batch["til_label"].to(device)
            survival_time = batch["survival_time"].to(device)

            til_pred, resp_pred, survival_pred = predict_with_multimodal_model(
                model=model, device=device,
                wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
            )

            if task == "response":
                mask = (response_label != -1)
                prob = torch.sigmoid(resp_pred[mask]).squeeze().cpu().numpy()
                label = response_label[mask].cpu().numpy()
                all_probs.extend(prob.tolist())
                all_labels.extend(label.tolist())
                all_ids.extend([i for i, m in zip(ids, mask) if m.item()])
            elif task == "til":
                mask = (til_label.sum(dim=1) >= 0)
                prob = torch.sigmoid(til_pred[mask]).cpu().numpy()
                label = til_label[mask].cpu().numpy()
                all_probs.extend(prob.tolist())
                all_labels.extend(label.tolist())
                all_ids.extend([i for i, m in zip(ids, mask) if m.item()])
            elif task == "survival":
                mask = (survival_time != -1)
                prob = survival_pred[mask].squeeze().cpu().numpy()
                label = survival_time[mask].cpu().numpy()
                all_probs.extend(prob.tolist())
                all_labels.extend(label.tolist())
                all_ids.extend([i for i, m in zip(ids, mask) if m.item()])
            else:
                raise ValueError(f"Unknown task for multimodal model: {task}")

    return pd.DataFrame({"id": all_ids, "label": all_labels, "prob": all_probs})

def predict_with_multimodal_model(
    model: torch.nn.Module,
    device: torch.device,
    wsi_feat: torch.Tensor = None,
    rna: torch.Tensor = None,
    methyl: torch.Tensor = None,
    prot: torch.Tensor = None,
    mirna: torch.Tensor = None
):
    """
    Performs a forward pass (prediction) using the ImmunovaMultimodalModel.

    Args:
        model (torch.nn.Module): The loaded ImmunovaMultimodalModel.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        wsi_feat (torch.Tensor, optional): WSI features tensor. Defaults to None.
        rna (torch.Tensor, optional): RNA omics tensor. Defaults to None.
        methyl (torch.Tensor, optional): Methylation omics tensor. Defaults to None.
        prot (torch.Tensor, optional): Protein omics tensor. Defaults to None.
        mirna (torch.Tensor, optional): miRNA omics tensor. Defaults to None.

    Returns:
        tuple: A tuple containing (til_pred, resp_pred, survival_pred) tensors.
    """
    model.eval()
    with torch.no_grad():
        til_pred, resp_pred, survival_pred = model(
            wsi_feat=wsi_feat.to(device) if wsi_feat is not None else None,
            rna=rna.to(device) if rna is not None else None,
            methyl=methyl.to(device) if methyl is not None else None,
            prot=prot.to(device) if prot is not None else None,
            mirna=mirna.to(device) if mirna is not None else None
        )
    return til_pred, resp_pred, survival_pred

def predict_survival_from_data(
    wsi_feat: torch.Tensor = None,
    rna: torch.Tensor = None,
    methyl: torch.Tensor = None,
    prot: torch.Tensor = None,
    mirna: torch.Tensor = None,
    model_path: str = "model.pth",
    input_dims: dict = None
):
    """
    Predicts survival using the ImmunovaMultimodalModel given WSI and omics data.

    Args:
        wsi_feat (torch.Tensor, optional): WSI features tensor (e.g., (1, N, 1024)). Defaults to None.
        rna (torch.Tensor, optional): RNA omics tensor (e.g., (1, 1000)). Defaults to None.
        methyl (torch.Tensor, optional): Methylation omics tensor (e.g., (1, 512)). Defaults to None.
        prot (torch.Tensor, optional): Protein omics tensor (e.g., (1, 256)). Defaults to None.
        mirna (torch.Tensor, optional): miRNA omics tensor (e.g., (1, 128)). Defaults to None.
        model_path (str): Path to the trained model weights (model.pth).
        input_dims (dict): Dictionary specifying input dimensions for omics data.

    Returns:
        torch.Tensor: The survival prediction tensor.
    """
    if input_dims is None:
        input_dims = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImmunovaMultimodalModel(input_dims=input_dims)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        _, _, survival_pred = model(
            wsi_feat=wsi_feat.to(device) if wsi_feat is not None else None,
            rna=rna.to(device) if rna is not None else None,
            methyl=methyl.to(device) if methyl is not None else None,
            prot=prot.to(device) if prot is not None else None,
            mirna=mirna.to(device) if mirna is not None else None
        )
    return survival_pred

if __name__ == '__main__':
    # Example usage for the multimodal model
    print("Calling multimodal model for 'response' task...")
    try:
        predictions_df = call_model(
            model_type='multimodal',
            task='response',
            model_path='model.pth',
            wsi_feature_path='wsi_feature/wsi_feature.pkl',
            omics_feature_path='omics_feature/omics_dict.pkl',
            label_feature_path='label_feature/label_dict_ACC.pkl',
            val_ids_path='val_ids.txt',
            input_dims={"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128},
            batch_size=32
        )
        print("Multimodal model predictions (first 5 rows):")
        print(predictions_df.head())
        predictions_df.to_csv("multimodal_predictions_response.csv", index=False)
        print("Predictions saved to multimodal_predictions_response.csv")

    except Exception as e:
        print(f"Error calling multimodal model: {e}")

    # You can add examples for other model types here once implemented
    print("\nCalling TIL CNN model...")
    try:
        til_predictions = call_model(
            model_type='til_cnn',
            model_path='Immunova2_Module1/til_model.pth',
            h5_file_path='Immunova2_Module1/acc-h5/combined.h5'
        )
        print("TIL CNN model predictions (first 5 rows):")
        print(til_predictions.head())
        til_predictions.to_csv("til_cnn_predictions.csv", index=False)
        print("Predictions saved to til_cnn_predictions.csv")
    except Exception as e:
        print(f"Error calling TIL CNN model: {e}")

    print("\nCalling DeepSurv model...")
    try:
        deepsurv_predictions = call_model(
            model_type='deepsurv',
            model_path='Immunova2_module2/trained_deepsurv_model.pt',
            data_path='Immunova2_module2/ALL_TCGA.csv'
        )
        print("DeepSurv model predictions (first 5 rows):")
        print(deepsurv_predictions.head())
        deepsurv_predictions.to_csv("deepsurv_predictions.csv", index=False)
        print("Predictions saved to deepsurv_predictions.csv")
    except Exception as e:
        print(f"Error calling DeepSurv model: {e}")

    print("\nCalling predict_survival_from_data with dummy data...")
    try:
        # Create dummy data (replace with your actual data)
        dummy_wsi_feat = torch.randn(1, 10, 512)  # Example: 1 sample, 10 patches, 512 features per patch
        dummy_rna = torch.randn(1, 1000)
        dummy_methyl = torch.randn(1, 512)
        dummy_prot = torch.randn(1, 256)
        dummy_mirna = torch.randn(1, 128)

        survival_prediction = predict_survival_from_data(
            wsi_feat=dummy_wsi_feat,
            rna=dummy_rna,
            methyl=dummy_methyl,
            prot=dummy_prot,
            mirna=dummy_mirna
        )
        print("Survival prediction:", survival_prediction.item())
    except Exception as e:
        print(f"Error predicting survival from data: {e}")
