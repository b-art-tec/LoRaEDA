import streamlit as st

st.image("assets/lora_dark.svg")

description = """
### Motivation.

Probabilistic modelling of network traffic can support both anomaly detection and synthetic data generation. Such models can help to identify transmissions that deviate from expected behaviour, while also generating realistic samples for testing monitoring and  components analysis.

This document presents an exploratory data analysis of the Medellin LoRaWAN dataset and an initial feature-engineering stage, as a preliminary step towards a complete modelling pipeline. The main focus is to refine problem representation.

The dataset provides a strong basis for our work because as it was collected over four months in a real urban LoRaWAN deployment. The available variables describe several aspects of communication, including timing and device information, geometric and link-budget parameters, transmission settings, environmental conditions, and channel indicators such as RSSI, SNR, and time on air. There are also other features of this dataset, that are important for us, that will be discussed in following sections.



"""

st.markdown(description)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("assets/anomaly_dark.svg", width=500)
