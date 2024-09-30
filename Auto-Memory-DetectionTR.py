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
    epsilon = 0.1  # Batch Size Artış Miktarı

    while not found_max_batch_size and batch_size <= max_batch_size:
        try:
            # Geçici bir data_loader oluşturun
            data_loader.batch_size = int(batch_size)

            start_time = time.time()

            # Eğitim döngüsü başlat
            for batch in data_loader:
                inputs, labels = batch
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                
                optimizer.zero_grad()

                # FP16 modunda eğitim
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass ve optimizer
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Batch size {int(batch_size)} başarılı. Geçen süre: {elapsed_time:.2f} saniye")

            # Daha verimli batch size kontrolü
            if elapsed_time < best_time:
                best_time = elapsed_time
                stable_count = 0
            else:
                stable_count += 1

            if stable_count >= patience:
                print(f"Performans iyileştirmesi durdu. Maksimum verimli batch size: {int(batch_size - epsilon)}")
                found_max_batch_size = True
                break

            batch_size += epsilon  # Batch size

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM hatasıyla karşılaşıldı. Maksimum batch size: {int(batch_size - epsilon)}")
                found_max_batch_size = True
                batch_size = batch_size - epsilon  # Bir önceki batch size en büyük boyut
                torch.cuda.empty_cache()  # GPU belleğini temizle
            else:
                raise e

        finally:
            # GPU belleğini temizleme Fonks.
            torch.cuda.empty_cache()

    return int(batch_size)

# Örnek kullanım:
model = model.to("cuda")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# İlk data_loader batch size'ı
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# Maksimum batch size'ı bul
max_batch_size = find_max_batch_size(model, data_loader, criterion, optimizer)
print(f"Bulunan maksimum batch size: {max_batch_size}")
