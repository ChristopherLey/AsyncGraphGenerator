# Asynchronous Graph Generator
## Init repository
to replicate the `conda` environment used for this code run
~~~console
conda env create -f environment.yaml
~~~
You'll need to set up a local mongoDB to store all the pre-processing for rapid read-write. Once you've set that up the
mongoDB config file can be found in the datareader paths e.g. `Datasets/Beijing/data/mongo_config.yaml`

config files take the form
~~~yaml
data_root: <data base root>
base: 'Activity'
host: '127.0.0.1'
port: 27017
~~~
## Run Activity training
To train the AGG using the activity dataset you can run
~~~console
python train_activity.py -c ./activity_config.yaml
~~~
all hyperparameters should be changed in the corresponding `activity_config.yaml`
schema file.

Similarly, for the __PM2.5__ dataset (Beijing) with the run file names `train_pm25.py` and the corresponding config file
called `pm25_config.yaml`

## AGG Conceptually
<table>
    <tr>
        <td> <img src="AGG_diagrams/time_series_matrix.png" width="200" title="Time Series Matrix"></td>
        <td><img src="AGG_diagrams/time_series_graph.png" width="200" title="Time series graph representation" ></td>
        <td><img src="AGG_diagrams/time_series_inputation.png" width="200" title="Node generation"></td>
    </tr>
    <tr>
        <td><p>(a) Matrix time-series representation</p></td>
        <td><p style="width: 200px; word-wrap: anywhere">(b) An asynchronous directed graph representing the sparse time series data with minimal assumptions other than the causal nature of the samples.</p></td>
        <td><p style="width: 200px">(c) New nodes can be generated arbitrarily using the encoded features of the asynchronous graph</p></td>
    </tr>
</table>

### Comparison
https://github.com/zjuwuyy-DL/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation/tree/main
