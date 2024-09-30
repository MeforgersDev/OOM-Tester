import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.cuda.amp import GradScaler, autocast

def find_max_batch_size(model, data_loader, criterion, optimizer, initial_batch_size=16, max_batch_size=1024, patience=3, increment=0.1):
    batch_size = initial_batch_size
    found_max_batch_size = False
    scaler = GradScaler()
    best_time = float('inf')
    stable_count = 0
    epsilon = 0.1 

    while not found_max_batch_size and batch_size <= max_batch_size:
        try:
            # Create a temporary data_loader
            data_loader.batch_size = int(batch_size)

            start_time = time.time()

            # Start training loop
            for batch in data_loader:
                inputs, labels = batch
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                
                optimizer.zero_grad()

                # Training in FP16 mode
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass and optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Batch size {int(batch_size)} successful. Time taken: {elapsed_time:.2f} seconds")

            # Check if there's a more efficient batch size
            if elapsed_time < best_time:
                best_time = elapsed_time
                stable_count = 0
            else:
                stable_count += 1

            if stable_count >= patience:
                print(f"Performance improvement stopped. Maximum efficient batch size: {int(batch_size - epsilon)}")
                found_max_batch_size = True
                break

            batch_size += epsilon 

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Encountered OOM error. Maximum batch size: {int(batch_size - epsilon)}")
                found_max_batch_size = True
                batch_size = batch_size - epsilon  # Previous batch size is the largest possible size
                torch.cuda.empty_cache()  # Clear GPU memory
            else:
                raise e

        finally:
            # Clear GPU memory after every attempt
            torch.cuda.empty_cache()

    return int(batch_size)

# Example usage:
model = model.to("cuda")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# Find the maximum batch size
max_batch_size = find_max_batch_size(model, data_loader, criterion, optimizer)
print(f"Maximum batch size found: {max_batch_size}")
