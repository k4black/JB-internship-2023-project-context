from pathlib import Path

import typer
from datasets import Dataset, DatasetDict
import numpy as np
from tqdm import tqdm


ROOT_FOLDER = Path(__file__).parent


app = typer.Typer(add_completion=False)


def _get_split_generator(dataset_path, split):
    def _gen():
        with open(dataset_path / f'java-small.{split}.c2s', 'r') as f:
            for line in tqdm(f.readlines(), desc=f'Processing {split}'):
                method_name, code = line.split(' ', 1)
                yield {
                    'label': method_name.replace('|', ' '),
                    'code': code.strip().replace('|', ' ').replace(',', ' ELEM_SEP '),
                }
    return _gen


@app.command()
def main(
        dataset_path: Path = typer.Option(ROOT_FOLDER / 'java-small-preprocessed', help='Path to processed dataset with train, test, validation sets'),
        dataset_name: str = typer.Option('k4black/code2seq-java-small-name-prediction', help='Dataset name to save'),
        push_to_hub: bool = typer.Option(False, help='Push model to HuggingFace Hub'),
):
    dataset = DatasetDict({
        split: Dataset.from_generator(_get_split_generator(dataset_path, split))
        for split in ['train', 'val', 'test']
    })

    for split in ['train', 'val', 'test']:
        print(split)
        ds = dataset[split]
        labels = [len(i.split()) for i in ds['label']]
        print('labels', np.mean(labels), max(labels))

    if push_to_hub:
        dataset.push_to_hub(dataset_name, private=True)


if __name__ == '__main__':
    app()
