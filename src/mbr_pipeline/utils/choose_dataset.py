from datasets import load_dataset

from src.mbr_pipeline.args import Args

DatasetArgs = Args.DatasetArgs


def get_dataset(
    dataset: DatasetArgs.SupportedDataset,
    split: DatasetArgs.DataSplit,
    end_index=-1,
    start_index=0,
    shuffle=False,
    seed=1,
):
    full_data = load_dataset(*dataset.value, split=split.value)
    # TODO: move input/output under standardized names

    # match dataset.name:
    if dataset.name == DatasetArgs.SupportedDataset.samsum.name:
        full_data = full_data.rename_column("dialogue", "input")
        full_data = full_data.rename_column("summary", "output")
    elif dataset.name == DatasetArgs.SupportedDataset.cnndm.name:
        full_data = full_data.rename_column("article", "input")
        full_data = full_data.rename_column("highlights", "output")
    elif dataset.name == DatasetArgs.SupportedDataset.xsum.name:
        full_data = full_data.rename_column("document", "input")
        full_data = full_data.rename_column("summary", "output")

    if shuffle:
        full_data = full_data.shuffle(seed=seed)

    if end_index != -1:
        return full_data.select(range(start_index, end_index))
    elif start_index != 0:
        return full_data.select(range(start_index, len(full_data)))
    else:
        return full_data


def test():
    args = DatasetArgs(
        dataset=DatasetArgs.SupportedDataset.samsum,
        split=DatasetArgs.DataSplit.test,
        end_index=351,
    )
    print(get_dataset(**args.__dict__))


# test()
