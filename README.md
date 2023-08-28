# CIFAR workshop on imablanced datasets

This repo contains the code to begin a workshop on CIFAR dataset with imbalanced classes.

To run it :

To install the dependencies for this project, please run the following command in your terminal:

    - if you have cuda core in your computer :
```bash
pip install -r requirements_cuda.txt
```

    - if you don't have cuda core in your computer :
```bash
pip install -r requirements.txt
```

Then :
```bash
python3 generate.py
```

This script will download CIFAR dataset and remove some data to make it imbalanced, by default class 3 is selected to be 1/10th of other classes.
It will also generate test_data.pkl which is required for evaluation of this project.

Then if you have cuda core and you don't want to waste time on training, you can directly run the jupyter notebook ```check.ipynb``` to generate output.pkl with the best model that we made for this project.

Otherwise, if you don't have cuda core or if you want to train the model with your computer, you have to run the jupyter notebook ```train_weight.ipynb``` to generate the best model of this project, then the jupyter notebook ```check.ipynb``` to generate the output.pkl for this project.

You can also run ```train.ipynb```(and respectively ```RA_train.ipynb``` or ```RA_train_weight.ipynb```) to generate model from other method. And to generate the output depending on these method, you have to change the checkpoint filename from ```checkpoint = torch.load('./models/weight_best_model.pth')``` to ```checkpoint = torch.load('./models/best_model.pth')``` (and respectively ```checkpoint = torch.load('./models/RA_best_model.pth')``` or ```checkpoint = torch.load('./models/RA_train_best_model.pth')```) in ```check.ipynb``` before your run it to get the output of theses methods.