from datasets import load_dataset

from args import Args

DatasetArgs = Args.DatasetArgs

def get_dataset(dataset: DatasetArgs.SupportedDataset, split: DatasetArgs.DataSplit, end_index=-1, start_index=0):
    full_data = load_dataset(*dataset.value, split=split.value)
    if end_index != -1:
        return full_data.select(range(start_index, end_index))
    elif start_index != 0:
        return full_data.select(range(start_index, len(full_data)))
    else:
        return full_data

def test():
    args = DatasetArgs(dataset=DatasetArgs.SupportedDataset.samsum, split=DatasetArgs.DataSplit.test, end_index=351)
    print(get_dataset(**args.__dict__))


#test()
