# Adaptive-TML
This repository provides the code of the "Incremental On-Device Tiny Machine Learning" paper and its extension "Tiny Machine Learning for Concept Drift" (under review).

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
The citation of the paper will be added soon.

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

## How to use

