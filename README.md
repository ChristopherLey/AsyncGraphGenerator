# Asynchronous Graph Generator
This is a repository for the code used in the paper ["Asynchronous Graph Generator"](https://arxiv.org/abs/2309.17335) by the authors

[Dr. Christopher P. Ley](https://www.linkedin.com/in/christopher-p-ley/) & Dr. Felipe Tobar.

## Abstract
We introduce the asynchronous graph generator (AGG), a novel graph attention network for imputation and prediction of
multi-channel time series. Free from recurrent components or assumptions about temporal/spatial regularity,
AGG encodes measurements, timestamps and channel-specific features directly in the nodes via learnable embeddings.
Through an attention mechanism, these embeddings allow for discovering expressive relationships among the variables of
interest in the form of a homogeneous graph. Once trained, AGG performs imputation by
_**conditional attention generation**_, i.e., by creating a new node conditioned on given timestamps and channel
specification. The proposed AGG is compared to related methods in the literature and its performance is analysed from a
data augmentation perspective. Our experiments reveal that AGG achieved state-of-the-art results in time series
imputation, classification and prediction for the benchmark datasets _Beijing Air Quality_,
_PhysioNet ICU 2012_ and _UCI localisation_, outperforming other recent attention-based networks.

## Conceptual Architecture
<img src="AGG_diagrams/data_preparation.png" width="400" title="Data Preparation">

An illustration of the AGG self-supervised pipeline.

**a)** Time series samples are collected (possibly) asynchronously, and comprise measurements, timestamps and channel features.

**b)** Samples are ordered and a split into inputs and targets for self-supervised training.

**c)** The input/target split is considered as instances of the asynchronous graph for training.

**d)** The learnt graph encodes a rich representation of the underlying signal, where new samples to be generated through
_**conditional attention generation**_; here, $c_g = blue$ and $\tau_g = t_N - t_*$.

### Node generation (Imputation & Prediction)
As previously mentioned, the AGG constructs asynchronous graphs of measurements free from assumptions about position and sample regularity.

A classical example of data with missing entries "forced" into synchronous form, or alternatively imposing implicit structure where it may not exists.

<img src="AGG_diagrams/time_series_matrix.png" width="200" title="Time Series graph">

Noteable assumptions about this form $x_2$ on channel 2 ($x_2 = 107$) happens at the same time as $x_2 = 17$ (channel 3). As we all as there is an order in channels, i.e. channel 1 proceeds channel 2.

The AGG relaxes spurious assumptions of structure and instead considers the multi-channel time series as a directed graph:

<img src="AGG_diagrams/time_series_graph.png" width="400" title="Time Series graph">

The directed graph encodes relevant causal relations and differences in channel and nothing more.
Relations (graph weighting) is learned through the graph attention mechanism.

Once relationships have been encoded in their respective nodes, $h_n$ is this diagram, these encoded node vectors can be
used to generate new _unseen_ measurements through a novel transductive node generation, called
__conditional node generation__. This conditions time and channel information to _inform_ the new relationships of the generated node.

<img src="AGG_diagrams/time_series_inputation.png" width="400" title="Time Series imputation">

## Structure

In the paper "Asynchronous Graph Generator" a common architecture was used seen below.
This consists of 2 _encoding_ layers and 1 _generation layer_.

<img src="AGG_diagrams/architecture_v2.png" width="800" title="AGG Architecture">

The sections of the network are indicated at the top of the figure. Inputs and target are
represented as circles and squares respectively, fixed operations are denoted by white blocks and learnable
transformations in light grey blocks.

It should be noted that the hyperparameters used are not _optimal_ but rather functional, further study is required
in order to consider an optimal hyperparameter structure. Nor is it required that the model consists of only 2 layers,
in fact the architecture (and this code) permits a model arbitrarily deep.


## Init repository
to replicate the `conda` environment used for this code run
~~~console
conda env create -f environment.yaml
~~~
You'll need to set up a local mongoDB to store all the pre-processing for rapid read-write. Once you've set that up the
mongoDB config file can be found in the datareader paths e.g. `Datasets/Beijing/data/mongo_config.yaml`

config files for [mongo_config.yaml](Datasets/Beijing/data/mongo_config.yaml) take the form
~~~yaml
data_root: <data base root>
base: 'Beijing'
host: '127.0.0.1'
port: 27017
~~~

Note that data creation (preprocessing in to mongoDB) is done in the corresponding `datareader` modules for each dataset.
Depending on the configuration of the data augmentation, **_this will take a while!_**. Consider the sensitivity
analysis for the data augmentation [here](Datasets/Sensitivity_Analysis). As stride length gets smaller, the number of
samples generated increases (approximately) exponentially.

<img src="AGG_diagrams/stride_sensitivity.png" width="400" title="Size Augmentation">

Stride length indirectly determines the number of samples generated for each dataset conceptually stride works as
described in the following diagram

<img src="AGG_diagrams/stride_diagram.png" width="400" title="Stride Diagram">

<img src="AGG_diagrams/stride_construction.png" width="400" title="Stride Description">
