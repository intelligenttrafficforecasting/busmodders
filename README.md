# busmodders
This repository uses deep learning models for predicting speeds of public busses in the Valby area, Copenhagen, using bus GPS data from [Movia](https://www.moviatrafik.dk/).

# results
Main models and results are found in a [Jupyter](https://jupyter.org/) [notebook](../master/model/EVERYTHING.ipynb). To run the code the dataset provided by [Movia](https://www.moviatrafik.dk/) is needed.

# requirements/dependencies
- python>=3.6.0
## python packages
- jupyter>=4.4.0
- numpy>=1.14.0
- pandas>=0.23.4
- scipy>=1.1.0
- yaml>=3.13
- argparse>=1.1
- tensorflow>=1.5.0
- folium>=0.6.0
- branca>=0.3.1
- geopandas>=0.4.0
## git submodules
- [DCRNN FORK](https://github.com/intelligenttrafficforecasting/DCRNN)
