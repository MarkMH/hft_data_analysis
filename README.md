![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/banner_hft.png)

<p align="justify" style="text-align:justify"> 
This project deals with market design approaches for eliminating undesirable arbitrage opportunities in the financial market. To this end, different designs are tested in an artificial financial market in the laboratory.  
</p>

<p align="justify" style="text-align:justify"> 
The data for this project will be generated through the economic experiments described in the following working paper: https://www.econtribute.de/RePEc/ajk/ajkdps/ECONtribute_172_2022.pdf. The data will be made available online, in accordance with the European Research Council's open access policy, once the experiments are completed. For the present project, data from preliminary pilot experiments were used.  
</p>

<p align="justify" style="text-align:justify"> 
The data generation process is designed such that each event in the market generates an observation. In this process, a microsecond-level timestamp is recorded for each observation. <b>HFT_MT.py</b> provides the data visualization and analysis, as well as the code snapshot below, which describes the pre-processing. First, any event prior to market starts is dropped, as events between trading periods are also captured while participants are viewing their preliminary results, for example. Second, the timestamps must be transformed such that calculations can be performed. Third, the time elapsed between entries for a given subject, called "timedelta", must be computed to observe, for example, how long a subject's order was in the market prior to execution.    
</p>

![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/hft_preprocessing.png)

---

<p align="justify" style="text-align:justify"> 
The main code calls 2 functions found in python_wip. The first, <b>create_pdf</b>, takes a Python DataFrame as input and creates a corresponding PDF file. The PDF file is optimized for use in LaTeX for scientific writings. The inputs for this function are straightforward, note that MultiIndexLevel takes integers and helps if the column name is too large, i.e. it splits the column name across n lines.</p> 

![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/create_pdf.png)
![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/summary_ms_liq.png)


--- 

<p align="justify" style="text-align:justify"> 
The <b>lineplot_prices</b> function is optimized for plotting multiple price charts within one figure. Especially when the key, as well as the labels of the axes are identical. It is also designed to name the sub-graphs in ascending order.</p>

![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/lineplot_prices.png)
![](https://github.com/MarkMH/hft_data_analysis/blob/610b92388bec772b8e77c672ddb2d5eec674b3a5/images/p3_prices_all.png)

---

<p align="justify" style="text-align:justify"> 
Both functions are work in progress and are updated continuously, if you are interested in the current state of these functions, just contact me. I'm happy to share my code and always welcome comments.</p>

