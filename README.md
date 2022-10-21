# Stock_Price_Tomorrow?

This a small project of forecasting stock price of few listed Indian companies. 

## Description

The Nifty50 dataset from kaggle has been used for this project which has historical stock prices of 50 Indian companies. I have selected 5 companies randomly from this set and apply deep learning based forecasting method to predict the future prices.

## Method

Diffent companies had different start date depending upon when they were listed in Nifty50 and was consistent till 30th April 2021. The data has high variability as the timeframe also covers the pandemic year and many previous fluctions/corrections in the market. The code randomly selected Adani Ports, BPCL, Coal India, HDFC Bank and Tech Mahindra to train and predict future closing values.

The model uses a combination of CNN and LSTM along with Dense layers to make the predictions using 136k trainable parameters. The 80% of datapoints were used for training and remaining 20% were used for testing.

## Result

Since the model learns just on the historical value of the stock prices without considering other economical factors, the predictions are somewhat good with companies that have less variation in the test data. For example, the stock price of Coal India has been less fluctuating in the train data but in the test data it constantly declines whereas the model predicts that the price will go up. Whereas the stock price of BPCL has seen abrupt downfall previously and hence it is able to predict the same changes in test data.

Adani Ports : MAE of 16.250679
[Adani Ports](https://github.com/Ayush-Mi/Stock_Price_Tomorrow-/blob/main/images/results_ADANIPORTS.png)

BPCL : MAE of 52.8978
[BPCL](https://github.com/Ayush-Mi/Stock_Price_Tomorrow-/blob/main/images/results_BPCL.png)

Coal India : MAE of 
[Coal India](https://github.com/Ayush-Mi/Stock_Price_Tomorrow-/blob/main/images/results_COALINDIA.png)

HDFC Bank : MAE of
[HDFC Bank](https://github.com/Ayush-Mi/Stock_Price_Tomorrow-/blob/main/images/results_HDFCBANK.png)

Tech Mahindra : MAE of 78.01814
[Tech Mahindra](https://github.com/Ayush-Mi/Stock_Price_Tomorrow-/blob/main/images/results_TECHM.png)


## Future Works
Adding other economical and social factor can help make better predictions.
