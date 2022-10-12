# Guide to using DVC
(Thanks to Sireesh for writing this up!)


## Getting and using data/checkpoints

The data for this project is managed and versioned by [DVC](https://dvc.org), and it is stored in [this Google Drive folder](https://drive.google.com/drive/folders/1DZHfXBQe3RyO6crvw8w-OqwSkEZvtGOx). Data and checkpoints should be stored in the `data/` folder.

You can find instructions for installing DVC [here](https://dvc.org/doc/install). Once you have DVC installed, run `dvc pull` from the root of the repo. This will pull down all the files that have been checked into DVC thus far.

DVC works in a similar fashion to [git-lfs](https://git-lfs.github.com/):
 it stores pointers and metadata for your data in the git repository,
while the files live elsewhere (in this case, on Google Drive). As you
work with data, such as in [the DVC tutorial](https://dvc.org/doc/start/data-and-model-versioning), DVC will automatically add the files you have tracked with it to the `.gitignore` file, and add new `.dvc` files that track the metadata associated with those files.

### Sample Workflow

* **Pull data down** : run `dvc pull` to pull down the data file into the repository folder
* **Modify your data** : as you would without DVC, use, modify, and work with your data.
* **Add new/modified data to DVC** : using `dvc add ...` in a similar fashion to a `git add`, add your new or modified data files to DVC
* **Add the corresponding metadata to git** : Once the data file has been added to DVC, a corresponding `.dvc` file will have been created. Add or update this into git, then push.
* **Sync the locally updated DVC data with the remote** : finally, push the data itself up to Google Drive with the `dvc push` command.

tl;dr:

* dvc pull
* dvc add <data_file>
* git add/commit <data_file.dvc>
* git push
* dvc push
