from dataset import get_dataloaders
import os
from utils.train_utils import train_one_epoch, evaluate
from model import get_device, get_model
import torch.nn as nn
import torch.optim as optim
import torch

def main():
    
    ################
    # DATA LOADING #
    ################
    
    # Define root paths
    
    ROOT_PATH_X = "/home/smitra16/.cache/kagglehub/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/versions/9/pcam"
    ROOT_PATH_Y = "/home/smitra16/.cache/kagglehub/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/versions/9/Labels/Labels"
    
    # Generate train, test and validation paths
    
    train_x_path = os.path.join(ROOT_PATH_X, 'training_split.h5' )
    train_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_train_y.h5')

    val_x_path = os.path.join(ROOT_PATH_X, 'validation_split.h5')
    val_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_valid_y.h5')

    test_x_path = os.path.join(ROOT_PATH_X, 'test_split.h5')
    test_y_path = os.path.join(ROOT_PATH_Y, 'camelyonpatch_level_2_split_test_y.h5')

    batch_size = 128 # define batch size

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_x_path, train_y_path,
        val_x_path, val_y_path,
        test_x_path, test_y_path,
        batch_size=batch_size
    )
    
    # Check to see if the right data is loaded
    
    print("\n Checking loaded data . . .")
    
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Unique labels in batch: {labels.unique()}")
    print(f"Length of training set: {len(train_loader)}, length of test set: {len(test_loader)}, length of val set:{len(val_loader)}")
    
    #################
    # MODEL LOADING #
    #################
    
    
    print('\n -----------------')
    print("Now designing model. . .")
    
    device = get_device()
    model = get_model()
    model = model.to(device)
    
    print("Model successfully loaded\n --------------")
    
    #################
    # TRAINING LOOP #
    #################
    
    
    print('Beginning training . . .')
    
    num_epochs = 100
    learning_rate = 1e-3
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
        
    # Final evaluation on test set
    model.load_state_dict(torch.load('models/best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
if __name__ == "__main__":
    main()
