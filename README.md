# RISC-V SIMD extension for the AI workload

This project is a fork of [CFU-Playground](https://github.com/google/CFU-Playground) framework, that allows AI workload, running in [TFLM](https://github.com/tensorflow/tflite-micro), acceleration with FPGA and CFU for RISC-V architecture 

## Docs
This project is implementation for Deep learning AMR model inference acceleration with CFU for edge systems and [bachelor's thesis](/docs/Bachelor_Thesis_Pavlo_Hilei.pdf).


## Description
### Datasets
Dataset used in article and bachelor's:
- [RadioML 2016(A,B)](https://www.deepsig.ai/datasets)

Another dataset used in bachelor's:
- Based on the following matlab [generation code](https://www.mathworks.com/help/deeplearning/ug/modulation-classification-with-deep-learning.html). Matlab dataset was generated with SNR in ranges \[0:29\]dB and \[-30:29\]dB to compare models performance. 

There are code samples with [RadioML 2018](https://www.deepsig.ai/datasets) and [MIGOU-MOD](https://data.mendeley.com/datasets/fkwr8mzndr/1), but were not used in bachelor's or article

### Models architectures
Generally two models architecture were trained:
- Transformer Encoder
- CNN

Models vizualizations are shown in thesis in Figures 4.2 and 4.3

Different hyper parameters where fine-tuned:
**CNN**
- Filter size
- Model width (number of channels)
- Model depth

**Encoder**
- Filter size
- Depth (number of encoder layers)
- Width

Accelerated CNN from article:

![](/media/cnn_article.svg)

Accelerated CNN from bachelor's:

![](/media/cnn.svg)

Encoder from article (W=32, N=4):

![](/media/article_transformer.svg)

Encoder from bachelor's:

![](/media/transformer.svg)

### Accelerator
Accelerator for developed iteratively, as CFU-Playground ideology suggests. 
All modifications and evolution overall is described in thesis. Final design (for bachelors) does multiply-accumulate between input and filter. At the end, quantization is performed to fit accumulated value back to int8. Input buffer is ring buffer, number of elements accumulated per clock is configurable. Due to limitations of CFU, data must be copied into temporary buffer. This design corresponds to CFU_V5 in an article, even though CFU_V6 differs in logic (as described in paper), diagram is the same:

![](/media/cfu_v8_6.svg)


## Code description

### Requirements and setup
The same as in the original [CFU-Playground project](https://github.com/google/CFU-Playground)
### Models
- All of the code related to models training, dataset generation and creating plots is under [`/models`](/models/) directory.
- Some of the experiments are in notebooks under [`/models/experiments`](/models/experiments)
- Datasets are not uploaded to the repository, but expected to be places under [`/models/data`](/models/data/)
- Matlab generation code is under [`/models/data_generation`](/models/data_generation/)
- Plots generation scripts are under [`/models/plots/`](/models/plots/)

### Verilog -- accelerators
- Different accelerators verilog implementations are under [`/verilog/verilog_src/`](/verilog/verilog_src/). Note, their names do not correspond to names in bachelor's or paper. Correspondance:
  - paper: CFU_V1 - thesis: CFU_V4 - code: CFU_V9
  - paper: CFU_V2 - thesis: CFU_V5 - code: CFU_V11.2
  - paper: CFU_V3 - thesis: CFU_V6 - code: CFU_V12.2
  - paper: CFU_V4 - thesis: CFU_V7 - code: CFU_V13.2
  - paper: CFU_V5 - thesis: CFU_V8 - code: CFU_V14
  - paper: CFU_V6 - thesis: none - code: CFU_V16
- Corresponding CPU code that uses accelerators is under [`/acceleration_src/src/`](/acceleration_src/src/)
- After model is deployed with code from [`/models/deployment_tools.py`](/models/deployment_tools.py), project under [`proj`](/proj/) is created. Original CFU-Playground makefiles are modified to take sources and verilog code from directories mentioned above
- to run simulation: `cd proj/myproject && make renode-headless`
- to compile project with vivado: `cd proj/myproject && make prog`
- to load code to synthesized CPU: `cd proj/myproject && make load`
- in generated Makefile of project, you can add other models and turn on/off cycles per layer profiling, adding wall time measuring anchors, etc. 
- Measuring wall time code is awful, but here is the procedure:
  - Uncomment/add `DEFINES += ADD_MEASURE_TIME_ANCHORS` to project makefile
  - You run `make prog` and `make load`.
  - You enter model menu, so that you have to only press 'g' to start model test
  - you run `python3 utils/measure_time.py -i -v`  from project root
  - you exit console from window started by `make load` by double `CTRL+C`
  - you type 'g' and enter in terminal with measuring script
  - you should see measured time prints in seconds.
- Plots generation code is under [`/proj/reports/plot/`](/proj/reports/plot/)



### Reports
- Reports data for models in [`/models/experiments/`](/models/experiments/). Directory with model name -- results.json
- Reports data for profiling and power consumption and resource utilization for paper: [`/proj/reports/article_reports_cnn_small_v2_q_aware/`](/proj/reports/article_reports_cnn_small_v2_q_aware/)
