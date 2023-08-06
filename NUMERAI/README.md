Numerai is a data science competition where you build machine learning models to predict the stock market. <br>
<br>
Each row in the dataset corresponds to a **stock** at a specific point in time, represented by the era. The features are quantitative attributes (e.g P/E ratio) known about the stock at the time, and the target is a measure of stock market returns 20 days into the future.  <br>
<br>
My objective is to build machine learning models to predict the target. <br>
<br>
Submissions are scored against two main metrics: <br>
- Correlation (CORR): prediction's correlation to the target <br>
- True contribution (TC):  prediction's contribution to the hedge fund's returns <br>
Since the target is a measure of 20 day stock market returns, it takes 20 days for each submission to be scored.


Every business day, new live features are released which represent the current state of the stock market. Your job is to generate live predictions and submit them to Numerai. Here is an example of how we generate and upload live predictions in Python: <br>
<br>
```python
# Authenticate
napi = numerapi.NumerAPI("api-public-id", "api-secret-key")


current_round = napi.get_current_round()

\comment : Download latest live features
napi.download_dataset(f"v4.1/live_{current_round}.parquet")
live_data = pd.read_parquet(f"v4.1/live_{current_round}.parquet")
live_features = live_data[[f for f in live_data.columns if "feature" in f]]

\comment : Generate live predictions
live_predictions = model.predict(live_features)

\comment :  submission
submission = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
submission.to_csv(f"prediction_{current_round}.csv")

\comment : Upload submission 
napi.upload_predictions(f"prediction_{current_round}.csv", model_id="your-model-id")
```


