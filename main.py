import torch
from datasets import load_dataset
import numpy as np
from PIL import Image
import torch as th
from torch.utils.data import DataLoader

from simple_transformers.transformer_encoder import ModalityEncoder, ModalityEncoderDecoder


def classification_train_loop(model, dataset, input_key, use_pil_images=False):
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    loss_fn = th.nn.CrossEntropyLoss()
    for epoch in range(10):
        losses, accuracies = [], []
        for batch in dataloader:
            optimizer.zero_grad()

            if use_pil_images:
                model_input = [Image.fromarray(i).convert('RGB') for i in batch[input_key].numpy()]
            elif input_key == 'image' and len(batch[input_key].shape) == 3:
                model_input = batch[input_key].unsqueeze(1).numpy()
            else:
                model_input = batch[input_key]
            logits = model(model_input)
            _, preds = th.max(logits, dim=-1)
            loss = loss_fn(logits, batch['label'])
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            accuracies += [th.mean((preds == batch['label']).float()).item()]
        print(f'LOSS: {np.mean(losses)}, ACCURACY: {np.mean(accuracies)}')

def text_gen_train_loop(model, dataset):
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = th.nn.CrossEntropyLoss()
    for epoch in range(10):
        losses, accuracies = [], []
        in_string, out_string, gen_string = None, None, None
        for batch in dataloader:
            optimizer.zero_grad()
            logits, gen_strings = model(batch['translation']['en'], batch['translation']['fr'])
            in_string, out_string, gen_string = batch['translation']['en'][0], batch['translation']['fr'][0], gen_strings[0]
            _, preds = th.max(logits, dim=-1)
            labels = model.output_preprocessor.tokenizer(batch['translation']['fr'], padding=True, return_tensors='pt')['input_ids']
            batch_size, seq_len = labels.shape
            # TODO: Loss should be masked too
            loss = loss_fn(logits.reshape(batch_size * seq_len, -1), labels.reshape(batch_size * seq_len))
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            # accuracies += [th.mean((preds == batch['label']).float()).item()]
        print(f'LOSS: {np.mean(losses)}') #, ACCURACY: {np.mean(accuracies)}')
        print(in_string)
        print(out_string)
        print(gen_string)

def action_gen_train_loop(model, dataset):
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=dataset.collate_fn)
    loss_fn = th.nn.CrossEntropyLoss()
    for epoch in range(50):
        losses, accuracies = [], []
        true_actions, gen_actions = None, None
        for batch in dataloader:
            optimizer.zero_grad()
            logits, gen_actions = model(batch['init_state']['input'], batch['actions']['input'], None, batch['actions']['attention_mask'])
            logits = logits[:, :-1]
            true_actions = batch['actions']['input'][0]
            _, preds = th.max(logits, dim=-1)
            # TODO labels should no have sos token so that offset exsists
            labels = th.tensor(batch['actions']['input'], device=logits.device)
            batch_size, seq_len = labels.shape
            # TODO: Loss should be masked too
            loss = loss_fn(logits.reshape(batch_size * seq_len, -1), labels.reshape(batch_size * seq_len))
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            # accuracies += [th.mean((preds == batch['label']).float()).item()]
        print(f'LOSS: {np.mean(losses)}, ACCURACY: {np.mean(accuracies)}')
        print(true_actions)
        print(gen_actions[0])

# MNIST, image classification
def run_mnist():
    use_pil_images = False
    dataset = load_dataset('mnist', split='train[:1000]')
    dataset.set_format(type='numpy', columns=['image', 'label'])
    dataset_kwargs = {'num_classes': 10, 'image_size': (224, 224), 'patch_size': (16, 16), 'num_channels': 3} \
                     if use_pil_images else \
                     {'num_classes': 10, 'image_size': (28, 28), 'patch_size': (7, 7), 'num_channels': 1}
    model = ModalityEncoder('images', **dataset_kwargs)
    classification_train_loop(model, dataset, 'image', use_pil_images=use_pil_images)

# SST2, text classification
def run_sst2():
    dataset = load_dataset('sst2', split='train[:1000]')
    dataset.set_format(type=None, columns=['sentence', 'label'])
    dataset_kwargs = {'num_classes': 2, 'max_text_length': 128}
    model = ModalityEncoder('text', **dataset_kwargs)
    classification_train_loop(model, dataset, 'sentence')

# Small WMT-En-Fr, translation
def run_translation():
    dataset = load_dataset('opus100', 'en-fr', split='test[:100]')
    # dataset.set_format(type=None, columns='translation')
    print(dataset[0])
    dataset_kwargs = {'max_text_length': 256}
    model = ModalityEncoderDecoder('text', 'text', **dataset_kwargs)
    text_gen_train_loop(model, dataset)

# Describe trajectories
def run_traj():
    from test_datasets.gscan_test_dataset import gSCAN
    from pathlib import Path
    dataset = gSCAN(Path('test_datasets'), 'test', gen_action=True)
    dataset_kwargs = dataset.get_kwargs()
    model = ModalityEncoderDecoder('init_state', 'actions', **dataset_kwargs)
    action_gen_train_loop(model, dataset)

run_mnist()