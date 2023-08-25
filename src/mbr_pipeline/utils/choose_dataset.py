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
    if not dataset.name.startswith("wmt"):
        full_data = load_dataset(*dataset.value, split=split.value)
    else:  # load translation datasets from disk
        # split="train" looks a bit sus, but it's because we only have one split
        # in our dataset file, which is named "train" by default
        full_data = load_dataset("json", data_files=dataset.value[1], split="train")

    if "id" not in full_data.features:
        full_data = full_data.add_column("id", list(range(len(full_data))))

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
    elif dataset.name == DatasetArgs.SupportedDataset.flores.name:
        full_data = full_data.rename_column("sentence_nob_Latn", "input")
        full_data = full_data.rename_column("sentence_eng_Latn", "output")
    elif dataset.name == DatasetArgs.SupportedDataset.flores_isl.name:
        full_data = full_data.rename_column("sentence_isl_Latn", "input")
        full_data = full_data.rename_column("sentence_eng_Latn", "output")
    elif dataset.name == DatasetArgs.SupportedDataset.wmt_en_de.name:
        full_data = full_data.rename_column("en", "input")
        full_data = full_data.rename_column("de", "output")
    elif dataset.name == DatasetArgs.SupportedDataset.wmt_ro_en.name:
        full_data = full_data.rename_column("ro", "input")
        full_data = full_data.rename_column("en", "output")

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
