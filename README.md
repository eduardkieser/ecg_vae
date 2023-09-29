# ECG Beat level anomaly detection
It's possible to use the data from 3 lead wearable ECG sensors to do individual beat level anomaly detection.  
What's more, it's possible to build this beat classifier using only publicly available PhysioNet data.  

## The approach
The central challenge of this project is the limited amount of annotated data. The annotations need to be made by experienced cardiologists and their time is a limited resource. In order to add value with a potentially limited model we will use it to develop a triage system in stead of a diagnostic system. The procedure is that the model identifies all the interesting beats that were identified in the 2 week recording period and presents them to a clinician in order of importance.

### Merge hierarchical labels where possible.  
The PhysioNet datasets to not use a standard form of beat anomaly, certain sets will identify the exact anomaly while others will only identify the family to which the anomaly belongs to. This means that data selection will be a balance between including as much data as possible and retaining as many annotations as possible.

### Boot strapping using unsupervised learning.  
Our first bit of ML involves training a VAE that will allow us to compress each beat to a latent space of around 5ish variables.

We then train a model that classifies the beat pathology based on the latent space values of a particular beat. That means that the eventual model will use the encoder of the VAE, strapped to the second classification model.

### VAE architecture
SISO v MIMO. It's interesting to stack different beats from different leads together, so if a single beat is a array with a length of 300 then the input/output shape will be 300 x n_leads.  
Using a 1D convolution on the encoder is a maybe, I think on the decoder this may be more tricky.