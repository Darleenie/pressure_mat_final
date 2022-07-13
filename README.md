# Intelligent and Efficient Pressure Sensing System with Hyper-Dimensional Computing

## Motivation

Indoor activity recognition for smart homes is enhanced by various sensor techniques and the recent development of machine learning algorithms. Existing pressure sensing systems usually employ a high-resolution but expensive mat, and use computationally intensive models like Convolutional Neural Networks (CNNs). 


## Method and results

In this paper, we propose an intelligent pressure sensing system including the hardware component of a low-cost and sufficiently accurate pressure mat using Velostat and the software component of Hyper-Dimensional Computing-based classification algorithm. Our system shows comparable accuracy with CNNs while consuming 85% less energy.


## Repository overview
```bash
├── README.md
├── data (download link below)
│   ├── static
│   └── time_series
├── lib
│   ├── dist
│   ├── torch-hd 
│   └── torch_hd.egg-info
└── src
    ├── CNN_main.py
    ├── CNN_time_series.py
    ├── HD.py 
    ├── HD_perm.py   
    ├── humoment.py  
    └── humoment_time_series.py
```


## Running instructions

You can download the dataset through this link
https://drive.google.com/drive/folders/1UwhnxmbkB4HpsYazrXF_sDSWjpyFHYSP?usp=sharing

Setup the environment, run:
```bash
pip install -r requirements.txt
```


Run static models in src:

HD:
```bash
python src/HD.py
```

CNN:
```bash
python src/CNN_main.py
```

Hu's Moments:
```bash
python src/humoment.py
```

Run time-series models in src:

HD:
```bash
python src/HD_perm.py
```

CNN:
```bash
python src/CNN_time_series.py
```

Hu's Moments:
```bash
python src/humoment_time_series.py
```
<!-- 
## More resources

Point interested users to any related literature and/or documentation.


## About

Explain who has contributed to the repository. You can say it has been part of a class you've taken at Tilburg University. -->
