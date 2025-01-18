## Auto Routing Station

### Description

1. Purpose: aim to automate the process of routing parcels everytime parcels arrives at Destination Hubs

2. Phases: divided into 3 phases

*   **Phase 1**: init ref data from MTB to Qdrant and run 1st ver of model
*   **Phase 2**: improve tool by using OPV2 API for realtime production
*   **Phase 3**: improve the model by adjusting wrong prediction data

### Installation
1. Clone the repository
2. Install the requirements
```bash
pip install -r requirements.txt
```

### Production
1. Qdrant: [**LINK**](https://5de6f86f-9666-4d5e-9b64-0718dbad2e4d.us-east4-0.gcp.cloud.qdrant.io:6333/dashboard#/collections) 
2. Folder: auto-routing/main.py
