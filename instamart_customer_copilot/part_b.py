import os, warnings, joblib
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
os.makedirs("outputs",exist_ok=True)
o=pd.read_csv("data/clean/orders.csv",parse_dates=["order_ts"]); c=pd.read_csv("data/clean/customers.csv",parse_dates=["signup_date"]); p=pd.read_csv("data/clean/products.csv")
done=o[o.status.eq("completed")].copy()
snap=pd.Timestamp("2025-10-01")
rfm=done.groupby("customer_id").agg(last=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum"))
rfm["recency"]=(snap-rfm.pop("last")).dt.days
rfm=rfm.reset_index()
sc=StandardScaler(); X=sc.fit_transform(rfm[["recency","frequency","monetary"]])
rows=[]
for k in range(2,9):
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
    rows.append((k,km.inertia_,silhouette_score(X,km.labels_)))
print("KMeans",rows)
K=4
km=KMeans(n_clusters=K,n_init=10,random_state=42).fit(X)
rfm["segment_id"]=km.labels_
prof=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).sort_values("monetary",ascending=False)
names={sid:("Premium Advocates" if i==0 else ["Occasional Buyers","Frequent Essentials","At-Risk Shoppers"][min(i-1,2)]) for i,sid in enumerate(prof.index)}
premium=prof.index[0]
rfm["segment_name"]=rfm.segment_id.map(names); rfm["is_premium"]=rfm.segment_id.eq(premium)
pca=PCA(n_components=2).fit(X); print("PCA",pca.explained_variance_ratio_)
rfm.to_csv("outputs/customer_segments.csv",index=False)
# Lapse cohort/features, strictly pre-cutoff
cut=pd.Timestamp("2025-09-01")
pre=o[(o.order_ts<cut)].copy()
cohort=pre[pre.status.eq("completed")].groupby("customer_id")
base=cohort.agg(last=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum"),avg_discount=("discount_pct","mean"),avg_delivery=("delivery_minutes","mean"))
base["recency"]=(cut-base.pop("last")).dt.days
allpre=pre.groupby("customer_id").agg(all_final=("order_id","nunique"),cancelled=("status",lambda s:(s=="cancelled").sum()))
base["cancel_rate"]=(allpre.cancelled/allpre.all_final).fillna(0)
fav=pre[pre.status.eq("completed")].merge(p[["product_id","category"]],on="product_id").groupby("customer_id").category.agg(lambda x:x.value_counts().index[0])
base["favourite_category"]=fav
base=base.join(c.set_index("customer_id")[["city","age","gender","signup_date"]])
base["tenure_days"]=(cut-base.signup_date).dt.days
post=done[(done.order_ts>=cut)&(done.order_ts<pd.Timestamp("2025-10-01"))].groupby("customer_id").order_id.nunique()
base["lapsed"]=(~base.index.isin(post.index)).astype(int)
base=base.reset_index()
features=["recency","frequency","monetary","avg_discount","cancel_rate","avg_delivery","favourite_category","city","age","gender","tenure_days"]
X=base[features]; y=base.lapsed
num=["recency","frequency","monetary","avg_discount","cancel_rate","avg_delivery","age","tenure_days"]; cat=["city","gender","favourite_category"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),cat)])
lr=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000))])
rf=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
gs1=GridSearchCV(lr,{"model__C":[.01,.1,1,10,100]},cv=5,scoring="f1",n_jobs=-1).fit(Xtr,ytr)
gs2=GridSearchCV(rf,{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]},cv=5,scoring="f1",n_jobs=-1).fit(Xtr,ytr)
for name,m in [("logistic",gs1.best_estimator_),("forest",gs2.best_estimator_)]:
    pred=m.predict(Xte); print(name, "best",gs1.best_params_ if name=="logistic" else gs2.best_params_,"cv",gs1.best_score_ if name=="logistic" else gs2.best_score_,"cm",confusion_matrix(yte,pred).tolist(),"P/R/F1",precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0))
# Majority-class baseline on the untouched test set.
baseline=np.zeros(len(yte),dtype=int)
print("baseline accuracy",float((baseline==yte).mean()),"lapsed precision/recall/F1",precision_score(yte,baseline,zero_division=0),recall_score(yte,baseline,zero_division=0),f1_score(yte,baseline,zero_division=0))
# Oversample minority class on training data only; test rows remain untouched.
train_df=Xtr.copy(); train_y=ytr.copy()
minor=train_df[ytr==1]
need=int((ytr==0).sum()-(ytr==1).sum())
if need>0:
    extra=minor.sample(n=need,replace=True,random_state=42)
    Xbal=pd.concat([train_df,extra]); ybal=pd.concat([train_y,pd.Series(np.ones(need,dtype=int),index=extra.index)])
    balanced_lr=gs1.best_estimator_; balanced_lr.fit(Xbal,ybal)
    bp=balanced_lr.predict(Xte)
    print("balanced logistic P/R/F1",precision_score(yte,bp,zero_division=0),recall_score(yte,bp,zero_division=0),f1_score(yte,bp,zero_division=0))
selected=gs2.best_estimator_ if f1_score(yte,gs2.best_estimator_.predict(Xte),zero_division=0)>=f1_score(yte,gs1.best_estimator_.predict(Xte),zero_division=0) else gs1.best_estimator_
joblib.dump(selected,"outputs/lapse_model.joblib"); loaded=joblib.load("outputs/lapse_model.joblib"); assert np.array_equal(selected.predict(Xte),loaded.predict(Xte))
# premium test customers: illustrative post-hoc flag
testids=set(base.loc[Xte.index,"customer_id"]) if False else set(base.loc[Xte.index,"customer_id"])
test=base[base.customer_id.isin(testids)].copy()
premium_ids=set(rfm.loc[rfm.is_premium,"customer_id"])
test=test[test.customer_id.isin(premium_ids)]
if len(test):
    probs=selected.predict_proba(test[features])[:,1]
    test["lapse_probability"]=probs
    test["actually_lapsed"]=test.lapsed
    test.sort_values("lapse_probability",ascending=False).head(10)[["customer_id","lapse_probability","actually_lapsed"]].to_csv("outputs/retention_list.csv",index=False)
else: pd.DataFrame(columns=["customer_id","lapse_probability","actually_lapsed"]).to_csv("outputs/retention_list.csv",index=False)
print("cohort",len(base),"lapsed",int(y.sum()),"train/test",len(Xtr),len(Xte))
