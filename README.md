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
1. Host: ```10.150.108.210```
2. Qdrant: [**LINK**](http://34.81.255.236:6333/dashboard#/collections) 
3. Cronjob: ``` 0,30 5-15 * * *```
4. Folder: auto-routing/main.py

### Others:
1. Dashboard: [**LINK**](https://lookerstudio.google.com/reporting/e0ec6492-ee57-4471-a344-20bd060a4941/page/S2cYE)

2. Data source:
*   rd_pending_data: [redash/2508](https://redash-vn.ninjavan.co/queries/2508)
*   rd_pending_rts: [redash/2501](https://redash-vn.ninjavan.co/queries/2501)
*   rd_rider_mapping: [redash/2563](https://redash-vn.ninjavan.co/queries/2563)
*   mtb_init_new_data: [metabase/103499](https://metabase.ninjavan.co/question/103499)
*   cloud_storage: [ml_auto_routing](https://console.cloud.google.com/storage/browser/vn-bi-storage/ml_auto_routing/pending_non_rts)
*   bigquery: [auto_routing_report](https://console.cloud.google.com/bigquery?project=vn-bi-product&ws=!1m4!1m3!3m2!1svn-bi-product!2sauto_routing_report)



