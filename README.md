# Adaptive-TML
This repository provides the code of the "Incremental On-Device Tiny Machine Learning" paper and its extension "Tiny Machine Learning for Concept Drift" (under review). The main contents are the five sklearn-based variants of the well-known k-Nearest Neighbors algorithm to the scenario of Tiny Machine Learning in presence of concept drift. Despite in this work the input of the kNN-based adaptive classifiers are assumed to receive in input the features extracted by an ad-hoc feature extractor (possibly followed by a dimensionality reduction operator), the implemented classes are general enough to be be executed with any input of a sklearn-based classifier.

The five proposed TML solutions are the following:
* Incremental kNN. A simple kNN that incrementally learns on each incoming sample.
* Condensing-base kNN. A kNN that relies on the condensing algorithm proposed by Hart in 1968 to reduce the memory and computational demands of the kNN algorithm.
* Condensing-in-Time kNN (or CIT or Passive Tiny kNN). A kNN that passively updates itself when the supervised information is provided by adding misclassfied samples to the kNN knowledge base.
* Active Tiny kNN. A kNN that employs a Change Detection Test to actively inspect for concept drft in the process generating the inputs. If a change is detected, then the kNN is adapted consequently.
* Hybrid Tiny kNN. A kNN that employs both the CIT and the Active Tiny kNN algorithms.

Please refer to the papers for details.

## Citation and Contact

If you use our work, please also cite the paper:
```
@inproceedings{disabato2020incremental,
  title={Incremental on-device tiny machine learning},
  author={Disabato, Simone and Roveri, Manuel},
  booktitle={Proceedings of the 2nd International Workshop on Challenges in Artificial Intelligence and Machine Learning for Internet of Things},
  pages={7--13},
  year={2020}
}
```
The citation of the other paper will be added soon.

## Abstract Incremental On-Device Tiny Machine Learning
> >  Tiny Machine Learning (TML) is a novel research area aiming at designing and developing Machine Learning (ML) techniques meant to be executed on Embedded Systems and Internet-of-Things (IoT) units. Such techniques, which take into account the constraints on computation, memory, and energy characterizing the hardware platform they operate on, exploit approximation and pruning mechanisms to reduce the computational load and the memory demand of Machine and Deep Learning (DL) algorithms.
> > Despite the advancement of the research, TML solutions present in the literature assume that Embedded Systems and IoT units support only the \textit{inference} of ML and DL algorithms, whereas their \textit{training}  is confined to more-powerful computing units (due to larger computational load and memory demand). This also prevents such pervasive devices from being able to learn in an incremental way directly from the field to improve the accuracy over time or to adapt to new working conditions.
> > The aim of this paper is to address such an open challenge by introducing an incremental algorithm based on transfer learning and k-nearest neighbor to support the on-device learning (and not only the inference) of ML and DL solutions on embedded systems and IoT units. Moreover, the proposed solution is general and can be applied to different application scenarios. %, while in this paper we focused on image classification and speech command. Experimental results on image/audio benchmarks and two off-the-shelf hardware platforms show the feasibility and effectiveness of the proposed solution.

## Abstract Tiny Machine Learning for Concept Drift
> > Tiny Machine Learning (TML) is a new research area whose goal is to design machine and deep learning techniques able to operate in Embedded Systems and IoT units, hence satisfying the severe technological constraints on memory, computation, and energy characterizing these pervasive devices. Interestingly, the related literature mainly focused on reducing the computational and memory demand of the inference phase of machine and deep learning models. At the same time, the training is typically assumed to be carried out in Cloud or edge computing systems (due to the larger memory and computational requirements). This assumption results in TML solutions that might become obsolete when the process generating the data is affected by concept drift (e.g., due to periodicity or seasonality effect, faults or malfunctioning affecting sensors or actuators, or changes in the users' behavior), a common situation in real-world application scenarios.  For the first time in the literature, this paper introduces a Tiny Machine Learning for Concept Drift (TML-CD) solution based on deep learning feature extractors and a k-nearest neighbors classifier integrating a hybrid adaptation module able to deal with concept drift affecting the data-generating process. This adaptation module continuously updates (in a passive way) the knowledge base of TML-CD and, at the same time, employs a Change Detection Test to inspect for changes (in an active way) to quickly adapt to concept drift by removing the obsolete knowledge. Experimental results on both image and audio benchmarks show the effectiveness of the proposed solution, whilst the porting of TML-CD on three off-the-shelf micro-controller units shows the feasibility of what is proposed in real-world pervasive systems.


## Installation
This code is written in `Python 3.8` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine in the desired directory:
```
git clone https://github.com/simdis/Adaptive-TML
```

To run the code, a virtual environment is highly suggested, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Adaptive-TML-directory>
virtualenv envname
source envname/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Adaptive-TML-directory>
conda create --name envname
source activate envname
while read requirement; do conda install -n envname --yes $requirement; done < requirements.txt
```

## Used Datasets.
The two papers have case studies in both image classification and speech command recognition scenarios. The employed datasets are:
* Image Classification dataset.
    * ImageNet
* Speech Command Identification Datasets
    * Speech Commands dataset (Warden, 2018).
    * Synthetic Speech Commands Dataset (Buchner, 2017), that extends the Speech Commands dataset.  

## How to use
The script named `main.py` is the python script able to run almost every possible configuration of both the papers. To configure each single experiment, you have to use the argparse flags, documented in the script. 

Please run
```
python main.py --help
```
to show all of them.

The most relevant ones are:
* --do_incremental, --do_passive, --do_active, --skip_base_exps, and --skip_nn_exps that allows to enable or disable the experiments with all the kNN variants.
* --data_dir and --second_data_dir. These two flags contain the path to the datasets (the first to the main one, the second, if provided, to the dataset after a change). Both the datasets are assumed to have the same set of classes and must contain a folder per each class. Use the flags --classes_to_change and --classes_to_add to further configure the post-change scenario.
* --dr_policy, --filters_to_keep, --class_distance_filters. These flags allow to define the dimensionality reduction operator. More details are available in both the papers.
* --cnn_fe, --cnn_fe_weights_path, --feature_layer. These flags allow to define the feature extractor. The path to the weights is required. More details are available in both the papers.
* --is_audio. The flag that switches between audio and image scenarios.

The default configuration (the same used in the paper "Tiny Machine Learning for Concept Drift") employs a feature extractor based on the first convolutional layer of the well-known ResNet-18 CNN pretrained on the ImageNet dataset and a dimensionality reduction operator that among the 64 filters of such layer keeps only the one having the highest activations on a baseline dataset. 

### Base experiment example
The following command allows to run the experiments on the non-incremental kNN algorithm, with a comparison to other baseline classifiers. In particular, the audio scenario is selected and 100 waveforms per class are provided as training set. 
```
python main.py [--seed <seed>] --dr_policy filter_selection_plus --filters_to_keep 51 --cnn_fe_weights_path <path_to_resnet18_pytorch_pth_file> --data_dir <first_audio_dataset_path> [--second_data_dir <second_audio_dataset_path>] [--output_dir <output_dir_path>] --is_audio --samples_per_class_to_test 100 --base_test_samples <num_samples_per_class_during_testing>
```

### Other experiments example
They will be added soon.
