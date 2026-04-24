import streamlit as st

st.set_page_config(
    page_title="Feature selection",
    layout="centered",
)

st.title("Feature selection")

st.markdown(r'''

### The rationale

In this section we focus on feature selection and engineering,  i.e. what we will include in the final training set, what will be modified, and what can be dropped. In other words, we need to decide what will count as a representation of the network state, or at least of those aspects of the state that matter for detection. By network state, we mean an abstraction of the available data that allows us to distinguish legitimate traffic from anomalies.


**The paradgim**

Here, we focus on the physical and hardware aspects of communication, so the state may be taken as a function of frame and gateway metadata, such as RSSI, SNR, frequency, or spreading factor. However, bare in mind that within MIRANDA and development of cyber-security digital twins this is only part of the challenge. The idea is that, if an impostor joins the network, the new device, being in an unusual location, may leave an unusual signature in the relation between the features we choose to include -- for example between declared coordinates and received signal strength. In LoRaWAN, many of these features are encoded in the metadata attached to transmissions, so the designer has plenty of freedom in choosing what aspect of the network will be modified.


**The strategy**

The current dataset is useful in this respect because it provides a rich set of timing, link-budget, environmental, and channel variables from a real deployment, which gives us many possible starting points for a representative model. At the same time, we should stay with quantities that are commonly used in practice. For example, quantities such as RSSI and SNR should be included explicitly, while some environmental conditions, such as humidity, are better treated implicitly, through uncertainty (unless they are part of the deployed system). A sensible way to approach this challenge is to start with key property that will pull, in some sense, all the features we need, which also we can later augment.


**Features related to propagation**

We start with the assertion, that simplest useful digital twin will have certain ability to predict network modifications. The most straightforward way to do it, is to expand it by adding another end device, of known hardware, at a new location. Therefore distance must enter the state description, together with RSSI. By definition,
$$
\mathrm{SNR} = \mathrm{RSSI} - P_n,
$$
where $$P_n$$ is the noise power. Therefore, by including also SNR we will improve our representation. For readers that are more ML focused, we remind that we are working here in logarithmic scales so ratios and proportions are represented by differences and sums respectively.

Will other features matter as well? Let us investigate this through a propagation model. A standard choice for narrow bands on sub-GHz frequencies, where changes in frequencies have relatively low impact on the link quality, is the log-distance path-loss model,
$$
\def \PL {\mathrm{PL}}
\PL(d) = \PL(d_0) + 10 n \log_{10} \left( \frac{d}{d_0} \right) + X_\sigma,
$$
where $\mathrm{PL}(d_0)$ is the path loss at a reference distance $$d_0$$, $$n$$ is the path-loss exponent, $d$ is the distance between transmitter and receiver, and $$X_\sigma$$ is a shadowing term, often $X_\sigma \sim \mathcal{N}(0, \sigma)$. So it seems that this segment of features is complete. We will expand on this later.


**TX/RX gains and losses**

The dataset includes detailed measurements of transmitter (TX) and receiver (RX) gains and losses. In practical deployments, these quantities are often not available and therefore should not form the core of a future generative model. However, our analysis suggests that they explain unusual variation observed in the dataset, for example in the relationship between distance and RSSI (you can explore it in the previous panel). For that reason, they are worth retaining in the present study as auxiliary variables. In future work, they may improve path-loss estimation or support a more informative _post hoc_ analysis.

By a similar argument, one could also consider antennas geometry and other relevant "physical" aspets of the system. However, incorporating such variables would require a more detailed propagation model, such as the one discussed in the following subsection. However, in the production setting the required data may be insufficient or unavailable.

**Feature engineering.**

It follows that the most convenient way to represent distance is on a logarithmic scale. This is consistent with the propagation models introduced above. Also, when we later derive the expected path loss, it may be more convenient, at least for prediction, to work with expected or estimated noise power rather than with SNR itself. In the context of this research, most of the features is in optimal representations simple because industry standards are already optimised for clarity and simplicity. For example, SNR and RSSI are almost always given in dB or dBm units.

**Contextual parameters**

Will other features matter as well? Let us consider the more advanced urban Okumura--Hata model

$$
\mathrm{PL}(\ldots) = A + B \log_{10}(f_c) - C \log_{10}(h_b) - a(h_m) + \left(D - E \log_{10}(h_b)\right)\log_{10}(d),
$$

where $$f_c$$ is the carrier frequency, $$h_b$$ is the base-station antenna height, and $$h_m$$ is the end-device antenna height. The correction term $$a(h_m)$$ accounts for the receiver antenna height and, in practice, depends on the urban setting. Capital letters represent to be estimated. In qualitative terms, the model says that path loss grows with distance and frequency, while higher antennas placements reduces it.

These are important considerations. It would therefore be tempting to include these parameters in the dataset.

However, such information is not usually available as part of the metadata. It is therefore better to treat it as a bias term in the path loss, and as one source of epistemic uncertainty. Frequency, on the other hand, appears in the model and is often reported. One could argue that, because the adaptive range is narrow, its main effect is again a bias term. Should it then be included? On the ground of satisfying propagation model, probably not, at least not in the context of this dataset. However, there are other reasons to include this aspect of communication.

**ADR policy and other features**

We can apply a very similar reasoning to other features as well. One of the key parameters defining the efficiency of a LoRa link is the spreading factor (SF). When adaptive data rate (ADR) is enabled, SF is usually adjusted according to the recent history of SNR, and as such, can be part of the location fingerprint. Any attempt to spoof a legitimate device and to communicate from another location, might result in anomalous relationship between SF, SNR and RSSI. 

With regard to other features related to transmission quality, such as detailed link-budget components (for example, transmission power and antenna gains) and environmental conditions, these are also typically unavailable. Accordingly, they should be treated as latent factors affecting the bias and variance of the observable radio metrics, in the same manner as was previously done for parameters such as antenna height.

The next consideration is related to the characterisation of the payload. While the analysis of its content belong to other layers of the digital twin, its volume might be an important feature allowing to detect anomalies.

Subsequently, quantities related to the data rate are also very relevant. The data rate is given by
$$
\text{R}_b = \text{SF} \left( \frac{4}{4 + \text{CR}} \right)\frac{\text{BW} \times 10^3}{2^{\text{SF}}},
$$

where $\text{R}_b$ denotes the nominal LoRa data rate in bit s$^{-1}$, $\text{SF}$ is the spreading factor, $\text{CR} \in \{1,2,3,4\}$ is the coding-rate parameter, and $\text{BW}$ is the signal bandwidth expressed in kHz. The factor $10^3$ converts the bandwidth from kHz to Hz so that $\text{R}_b$ is obtained in bits per second. Equivalently, the term $\text{BW}/2^{\text{SF}}$ represents the symbol rate, while $4/(4 + \text{CR})$ accounts for the reduction in effective information rate due to forward error correction.

While these parameters combined with time-on-air (ToA) provide indication of the payload, more importantly they also provide some level of verification if link adheres to the network policies. Therefore, if available they should also be a part of the feature set.

### Final training set (QoI)

Based on these considerations, we selected the following set of features.

| Feature        |               Symbol | Description                                                                                         |
| -------------- | -------------------: | --------------------------------------------------------------------------------------------------- |
| `timestamp`    |                $$t$$ | Measurement time (used for temporal context/splitting and drift).                                   |
| `device_id`    |      $$\mathrm{id}$$ | End-node identifier (captures per-device fingerprints and fixed deployment differences).            |
| `distance`     |                $$d$$ | EN–GW separation in metres (primary geometric driver of propagation/path loss).                     |
| `frequency`    |                $$f$$ | Carrier frequency (contextual PHY parameter; can shift propagation and channel conditions).         |
| `frame_length` | $$l_{\text{frame}}$$ | Payload size in bytes (behaviour/policy proxy without inspecting payload content).                  |
| `rssi`         |    $$\mathrm{RSSI}$$ | Received signal strength at the gateway in dBm (core observable of link power).                     |
| `snr`          |     $$\mathrm{SNR}$$ | Signal-to-noise ratio in dB (core observable of channel quality).                                   |
| `toa`          |     $$\mathrm{ToA}$$ | Time-on-air in seconds (captures airtime regime and potential policy violations).                   |
| `sf`           |      $$\mathrm{SF}$$ | Spreading factor (ADR-relevant parameter; relates to coverage/airtime trade-offs and fingerprints). |

Additinally, the training set will include TX and RX gains and losses (gtx, ltx, grx, lrx).
''')
