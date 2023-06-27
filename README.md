# SynthText for (English + Japanese)
Code for generating synthetic text images as described in ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) with support for japanese characters

## TODO

Add support for chinese

## Output samples


**Synthetic Japanese Text Samples 1**

![Japanese example 1](results/sample1.png "Synthetic Japanese Text Samples 1")


**Synthetic Japanese Text Samples 2**

![Japanese example 2](results/sample2.png "Synthetic Japanese Text Samples 2")


**Synthetic Japanese Text Samples 3**

![Japanese example 3](results/sample3.png "Synthetic Japanese Text Samples 3")


**Synthetic Japanese Text Samples 4**

![Japanese example 4](results/sample4.png "Synthetic Japanese Text Samples 4")


## Background
This repository is a fork of [jinmingteo/SynthText](https://github.com/jinmingteo/SynthText). Changes were made to resolve some dependency issues as well as to add some scripts that can help you generate your own custom dataset, rather than just using the small sample. Note that [jinmingteo/SynthText](https://github.com/jinmingteo/SynthText) was forked from [gachiemchiep/SynthText](https://github.com/gachiemchiep/SynthText) to fix other bugs, which in turn was forked from the original [ankush-me/SynthText](https://github.com/ankush-me/SynthText) in order to create a way to generate a Japanese version of the SynthText dataset. If you can't find what you are looking for here, it may be useful to refer to the other forks as well.

## The main differences
1. Use newer version of opencv
2. Use nltk to parse language (eng, jpn)

## How to use this source
### Preparation
Put your text data and font as follow
```
data
├── dset.h5
├── fonts
│   ├── fontlist.txt                        : your font list
│   ├── ubuntu
│   ├── ubuntucondensed
│   ├── ubuntujapanese                      : your japanese font
│   └── ubuntumono
├── models
│   ├── char_freq.cp
│   ├── colors_new.cp
│   └── font_px2pt.cp
└── newsgroup
    └── newsgroup.txt                       : your text source
```

### Install dependencies
```
# For japanese
sudo apt-get install libmecab2 libmecab-dev mecab mecab-ipadic mecab-ipadic-utf8 mecab-utils
```

### Prepare your python environment
You can use whatever python version management system you want, but I prefer to use [pyenv](https://github.com/pyenv/pyenv).

1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Install Python 3.6.5
* Note: Using Python 3.7+ will cause issues when trying to install pygame, so don't do that.
```bash
pyenv install 3.6.5
pyenv global 3.6.5
```
3. Create your virtual environment, and then activate it.
```bash
python -m pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
```
4. Install the python dependencies.
* Note: The code in this repo may stop working if you change the versions of any of the dependencies. In particular, newer versions of pygame have been confirmed to cause problems with the bounding box annotations.
```bash
python -m pip install -r requirements.txt
```

### Generate font model and char model
```
python invert_font_size.py
python update_freq.py

mv char_freq.cp data/models/
mv font_px2pt.cp data/models/
```

### Download the background data
You can find the background data used for creating the SynthText dataset in [this torrent](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c).

### Generate a sample
```bash
python create_sample.py --dataDir path/to/downloaded/dataDir --out where/you/want/to/save/sample
```

### Generate the dataset
Note: You may want to adjust some of the parameters in the script itself.
```bash
python gen.py --lang JPN
```

### Visualize Dataset
```bash
python read_results.py --datasetPath path/to/SynthText.h5 --visDump path/to/output/vis/dump
```

# SynthText
Code for generating synthetic text images as described in ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).


**Synthetic Scene-Text Image Samples**
![Synthetic Scene-Text Samples](samples.png "Synthetic Samples")

### Generating samples

```
python gen.py --viz --lang ENG/JPN
```

This will download a data file (~56M) to the `data` directory. This data file includes:

  - **dset.h5**: This is a sample h5 file which contains a set of 5 images along with their depth and segmentation information. Note, this is just given as an example; you are encouraged to add more images (along with their depth and segmentation information) to this database for your own use.
  - **data/fonts**: three sample fonts (add more fonts to this folder and then update `fonts/fontlist.txt` with their paths).
  - **data/newsgroup**: Text-source (from the News Group dataset). This can be subsituted with any text file. Look inside `text_utils.py` to see how the text inside this file is used by the renderer.
  - **data/models/colors_new.cp**: Color-model (foreground/background text color model), learnt from the IIIT-5K word dataset.
  - **data/models**: Other cPickle files (**char\_freq.cp**: frequency of each character in the text dataset; **font\_px2pt.cp**: conversion from pt to px for various fonts: If you add a new font, make sure that the corresponding model is present in this file, if not you can add it by adapting `invert_font_size.py`).

This script will generate random scene-text image samples and store them in an h5 file in `results/SynthText.h5`. If the `--viz` option is specified, the generated output will be visualized as the script is being run; omit the `--viz` option to turn-off the visualizations. If you want to visualize the results stored in  `results/SynthText.h5` later, run:

```
python visualize_results.py
```
### Pre-generated Dataset
A dataset with approximately 800000 synthetic scene-text images generated with this code can be found [here](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Adding New Images
Segmentation and depth-maps are required to use new images as background. Sample scripts for obtaining these are available [here](https://github.com/ankush-me/SynthText/tree/master/prep_scripts).

* `predict_depth.m` MATLAB script to regress a depth mask for a given RGB image; uses the network of [Liu etal.](https://bitbucket.org/fayao/dcnf-fcsp/) However, more recent works (e.g., [this](https://github.com/iro-cp/FCRN-DepthPrediction)) might give better results.
* `run_ucm.m` and `floodFill.py` for getting segmentation masks using [gPb-UCM](https://github.com/jponttuset/mcg).

For an explanation of the fields in `dset.h5` (e.g.: `seg`,`area`,`label`), please check this [comment](https://github.com/ankush-me/SynthText/issues/5#issuecomment-274490044).

### Pre-processed Background Images
The 8,000 background images used in the paper, along with their segmentation and depth masks, have been uploaded here:
`http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/<filename>`, where, `<filename>` can be:

|    filenames    | size |                      description                     |             md5 hash             |
|:--------------- | ----:|:---------------------------------------------------- |:-------------------------------- |
| `imnames.cp`    | 180K | names of images which do not contain background text |                                  |
| `bg_img.tar.gz` | 8.9G | images (filter these using `imnames.cp`)             | 3eac26af5f731792c9d95838a23b5047 |
| `depth.h5`      |  15G | depth maps                                           | af97f6e6c9651af4efb7b1ff12a5dc1b |
| `seg.h5`        | 6.9G | segmentation maps                                    | 1605f6e629b2524a3902a5ea729e86b2 |

Note: due to large size, `depth.h5` is also available for download as 3-part split-files of 5G each.
These part files are named: `depth.h5-00, depth.h5-01, depth.h5-02`. Download using the path above, and put them together using `cat depth.h5-0* > depth.h5`.

[`use_preproc_bg.py`](https://github.com/ankush-me/SynthText/blob/master/use_preproc_bg.py) provides sample code for reading this data.

Note: I do not own the copyright to these images.

### Generating Samples with Text in non-Latin (English) Scripts
- @JarveeLee has modified the pipeline for generating samples with Chinese text [here](https://github.com/JarveeLee/SynthText_Chinese_version).
- @adavoudi has modified it for arabic/persian script, which flows from right-to-left [here](https://github.com/adavoudi/SynthText).
- @MichalBusta has adapted it for a number of languages (e.g. Bangla, Arabic, Chinese, Japanese, Korean) [here](https://github.com/MichalBusta/E2E-MLT).
- @gachiemchiep has adapted for Japanese [here](https://github.com/gachiemchiep/SynthText).
- @gungui98 has adapted for Vietnamese [here](https://github.com/gungui98/SynthText).
- @youngkyung has adapted for Korean [here](https://github.com/youngkyung/SynthText_kr).

### Further Information
Please refer to the paper for more information, or contact me (email address in the paper).
