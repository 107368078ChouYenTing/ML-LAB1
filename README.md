# ML-LAB1
house predict

資料分析
===
找出最適合取樣數據的機率密度的係數
<pre>(mu, sigma) = norm.fit(train['price'])</code></pre>

提高計算上的有效位數
<pre>train["price"] = np.log1p(train["price"])</code></pre>

確認各項目的斜率
<pre>skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))</code></pre>

對於斜率過大的數據利用box-cox轉換函數以降低斜率
<pre>skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
</code></pre>

做法
===
使用基本模型
<pre>#LASSO Regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

#Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#Gradient Boosting Regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)

#XGBoost :
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state =7, nthread = -1)

#LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)</code></pre>

使用stacking model的方式將base model結合計算
<pre>#Simplest Stacking approach : Averaging base models
#Averaged base models class
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
#Averaged base models score
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))</code></pre>

權重分配(可自行調整)
<pre>ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15</code></pre>

結果分析
===
測試了兩種權重分配
<pre>ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15</code></pre>
<pre>ensemble = stacked_pred*0.80 + xgb_pred*0.10 + lgb_pred*0.10</code></pre>
後者較前者進步了200的誤差值
