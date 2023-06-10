import torch
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

def encoded_data(tokenizer, data):
    encoded_data = tokenizer.batch_encode_plus(
        data['sentence'].tolist(),
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return encoded_data

def create_dataset(encoded_data, data, device):
    dataset = TensorDataset(
        encoded_data['input_ids'].to(device),
        encoded_data['attention_mask'].to(device),
        torch.tensor(data['labels'].tolist()).to(device)
    )

    return dataset

def trainer(train_loss, val_losses, val_accuracies, num_epochs, train_loader, val_dataset, val_loader, model, optimizer, scheduler, device):
    # Training Electra model
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate the model
        model.eval()
        val_loss, val_acc, val_steps = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                logits = outputs[1]
                val_loss += loss.item()
                val_acc += (logits.argmax(1) == labels).sum().item()
                val_steps += 1

            avg_val_loss = val_loss / val_steps
            avg_val_acc = val_acc / len(val_dataset)
            
            train_loss.append(loss.item())
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            
        print("\n--------------------------------------------")
        print('Epoch {:} / {:}'.format(epoch + 1, num_epochs))
        print("Training loss: ", loss.item())
        print("Validation loss: ", avg_val_loss)
        print("Validation accuracy: ", avg_val_acc)

def testing(model, test_loader, test_data, device):
    start = time.time()
    model.eval()
    acc = 0
    test_loss = 0
    test_steps = 0

    with torch.no_grad(): 
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            test_loss += loss.item()
            acc += (logits.argmax(1) == labels).sum().item()
            test_steps += 1

        accuracy = acc / len(test_data)

    print("Test loss", test_loss / test_steps)
    print("Test accuracy: {:.2f}%".format(accuracy*100))
    print("Time",time.time()-start)

def print_layer(model):
    for name, param in model.named_parameters():
        if 'electra' or 'classifier' or 'pre_classifier' in name:
            print(f'{name}: {param.requires_grad}')

def unfreeze_last_layer(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_layer_last_block(model):
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_2block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_3block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_4block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True     

def unfreeze_last_layer_last_5block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True
    
    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True   

def unfreeze_last_layer_last_6block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True     

def unfreeze_last_layer_last_7block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_8block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[4].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_9block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[4].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[3].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_10block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[4].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[3].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[2].parameters():
        param.requires_grad = True

def unfreeze_last_layer_last_10block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[4].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[3].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[2].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[1].parameters():
        param.requires_grad = True    

def unfreeze_last_layer_last_11block(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[11].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[10].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[9].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[8].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[7].parameters():
        param.requires_grad = True 

    for param in model.electra.encoder.layer[6].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[5].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[4].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[3].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[2].parameters():
        param.requires_grad = True

    for param in model.electra.encoder.layer[1].parameters():
        param.requires_grad = True            

    for param in model.electra.encoder.layer[0].parameters():
        param.requires_grad = True