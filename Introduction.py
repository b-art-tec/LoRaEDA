import streamlit as st


st.set_page_config(page_title="LoRaWAN EDA", layout="centered")

st.image("assets/smart_city_baner.png")

motivation = """
### Cybersecurity Digital Twins and LoRaWAN in the context of Smart-Cities ... 

Cybersecurity Digital Twins (CSDTs) are digital representations of connected systems that combine information about assets, configurations, dependencies, and possible attack paths. In the context of the [MIRANDA project](https://www.mirandaproject.eu/), a CSDT is used to model the security posture of complex, heterogeneous, and multi-ownership ICT systems, so that operators can analyse risks, anticipate threats, and support detection and response in a proactive but non-invasive way.

CSDTs are particularly important in smart cities because urban services rely on many interconnected platforms, devices, and providers, often organised as evolving digital service chains. In such settings, limited visibility across organisational boundaries can make attacks harder to detect and contain, which makes a shared and adaptive security view especially valuable.

LPWAN stands for _Low-Power Wide-Area Network_. It refers to wireless technologies designed to connect devices over long distances while keeping energy consumption low. [LoRaWAN](https://lora-alliance.org/about-lorawan/) is an open LPWAN protocol for IoT systems that supports bidirectional communication, end-to-end security, mobility, and localisation services over the LoRa physical layer.

In this setting, LoRaWAN fits at the edge of the smart-city infrastructure, where sensors and actuators collect data from streets, buildings, transport systems, as well as utilities, and forward it to gateways and other services. It therefore forms part of the broader digital service chain that should be observed and protected.

Here, security can be improved by adding a first layer of defence based on the physical aspects of communication, such as RSSI, SNR, ADR, etc.. These features do not replace other protection mechanism. Instead, they add early context for anomaly detection, device verification, and location-consistency checks, which can help reveal suspicious behaviour before it affects other layers of the system.

"""

st.markdown(motivation)
