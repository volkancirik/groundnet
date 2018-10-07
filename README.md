# GroundNet
Repository for AAAI 2018 paper "[Using Syntax for Referring Expression Recognition](https://www.cs.cmu.edu/~vcirik/assets/pdf/AAAI2018_cirik_using.pdf)".

    @article{cirik2018using,
	title={Using Syntax to Ground Referring Expressions in Natural Images},
    author={Cirik, Volkan and Berg-Kirkpatrick, Taylor and Morency, Louis-Philippe},
    journal = {AAAI},
    year = {2018}
    }


### Preparing Environment:

The code is written in an old version of Pytorch. So, first create a virtual environment for your pytorch and other packages like h5py:

    conda create -n pytorch0.1.12 python=2.7
    source activate pytorch0.1.12
    conda install pytorch=0.1.12 h5py tqdm scipy pillow matplotlib scikit-image nltk
    conda install -c conda-forge ipython jupyter ipywidgets

    python -m ipykernel install --user --name=pytorch0.1.12
Compile [Google_Refex_toolbox](https://github.com/mjhucla/Google_Refexp_toolbox)

    cd Google_Refexp_toolbox; python setup.py
    
To be able to run the parser, you also need Java 8. Note that the first time you run a new installation of pytorch it takes some time to compile the cuda kernels, just be patient. 

  Now you need to download data I used for the experiments by running following commands.
  Under `data/`:

    bash download_coco.sh
    bash download_imdb.sh
    bash download_refcocog.sh
  Under `parser/`:
  
    bash download_parser.sh
  Under `notebooks/`:
  
    bash download_sup_ann.sh
  Under `downloaded_models/`:
  
    bash download_models.sh
    
### Training & Testing Model

In `exp-refcocog/` run the following command to list the options for training models 

    source activate pytorch0.1.12
    PYTHONPATH=.. python train.py --help
    
To test a trained model run as follows:

    source activate pytorch0.1.12
    PYTHONPATH=.. python test.py --resume ../downloaded_models/snapshot.groundnet.model --out-file groundnet.test.json

To evaluate a model for supporting object annotations:
    
    PYTHONPATH=.. python analyze.py --category sup_acc --prediction groundnet.test.json

### Visualizing Errors

Run your jupyter notebook under the root folder of the repo and don't forget to switch `pytorch0.1.12` kernel. Then simply run the `notebooks/interactiveModel.ipynb`. You should be able to enter a referring expression to test a model. To play with [CMN](https://github.com/ronghanghu/cmn) model, just uncomment the `args.resume` and `args.dump` line.

### Acknowledgements

I thank Licheng Yu, Ronghang Hu, Varun Nagaraja for their help.
