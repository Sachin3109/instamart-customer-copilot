import os, sqlite3, pandas as pd, numpy as np
from pathlib import Path
os.makedirs("data/clean",exist_ok=True); os.makedirs("data",exist_ok=True); os.makedirs("sql",exist_ok=True)
base="data/raw/"
c=pd.read_csv(base+"customers.csv",keep_default_na=False)
p=pd.read_csv(base+"products.csv")
o=pd.read_csv(base+"orders.csv",keep_default_na=False)
# Customer normalization
c["city"]=c.city.astype(str).str.strip().str.title()
c["gender"]=c.gender.astype(str).str.strip().str.lower().map({"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
c["age"]=pd.to_numeric(c.age,errors="coerce"); c.loc[~c.age.between(10,100),"age"]=np.nan
c["signup_date"]=pd.to_datetime(c.signup_date)
c["age"]=c["age"].fillna(c.groupby("city").age.transform("median")).round().astype(int)
# Orders clean in required order
o["order_ts"]=pd.to_datetime(o.order_ts)
o=o.drop_duplicates()
o=o.sort_values("order_ts").drop_duplicates("order_id",keep="last").copy()
for col in ["quantity","discount_pct","order_amount","delivery_minutes"]:
    o[col]=pd.to_numeric(o[col],errors="coerce")
o=o.merge(p[["product_id","price_inr","category"]],on="product_id",how="left")
missing=o.order_amount.isna()
o.loc[missing,"order_amount"]=(o.loc[missing,"price_inr"]*o.loc[missing,"quantity"]*(1-o.loc[missing,"discount_pct"]/100)).round(2)
o.loc[o.delivery_minutes<0,"delivery_minutes"]=np.nan
mask=o.status.eq("completed")
med=o.loc[mask].groupby("category").delivery_minutes.transform("median")
o.loc[mask,"delivery_minutes"]=o.loc[mask,"delivery_minutes"].fillna(med)
o=o.drop(columns=["price_inr","category"])
o["order_amount"]=o.order_amount.round(2)
# outputs
c.to_csv("data/clean/customers.csv",index=False)
p.to_csv("data/clean/products.csv",index=False)
o.to_csv("data/clean/orders.csv",index=False)
con=sqlite3.connect("data/instamart.db")
for name,df in [("customers",c),("products",p),("orders",o)]:
    x=df.copy()
    if name=="orders": x["order_ts"]=x.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    if name=="customers": x["signup_date"]=x.signup_date.dt.strftime("%Y-%m-%d")
    x.to_sql(name,con,if_exists="replace",index=False)
con.close()
queries = {
"Q1":"""SELECT customer_id, SUM(order_amount) total FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;""",
"Q2":"""SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;""",
"Q3":"""SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);""",
"Q4":"""SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;""",
"Q5": """WITH spend AS (SELECT customer_id,SUM(order_amount) total FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total) av FROM spend) SELECT customer_id,total FROM spend WHERE total>(SELECT av FROM avg_spend);""",
"Q6":"""WITH s AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) rn FROM s ORDER BY city,rnk;""",
"Q7":"""WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily ORDER BY day;"""
}
with open("sql/queries.sql","w") as f:
    for k,v in queries.items(): f.write("-- "+k+"\n"+v+"\n\n")
con=sqlite3.connect("data/instamart.db")
for k,q in queries.items():
    df=pd.read_sql_query(q,con); print(k,len(df),df.head(2).to_dict("records"))
con.close()
# checks
assert len(c)==1200 and c.city.nunique()==6 and c.age.notna().all()
assert len(o)==12309 and set(o.status)=={"completed","cancelled"}
assert o.loc[o.status.eq("completed"),"delivery_minutes"].notna().all()
assert o.loc[o.status.eq("cancelled"),"delivery_minutes"].isna().all()
assert round(o.loc[o.status.eq("completed"),"order_amount"].sum(),2)==2998668.80
assert len(pd.read_sql_query(queries["Q1"],sqlite3.connect("data/instamart.db")))==30
assert len(pd.read_sql_query(queries["Q2"],sqlite3.connect("data/instamart.db")))==30
assert len(pd.read_sql_query(queries["Q3"],sqlite3.connect("data/instamart.db")))==11
assert len(pd.read_sql_query(queries["Q4"],sqlite3.connect("data/instamart.db")))==25
assert len(pd.read_sql_query(queries["Q5"],sqlite3.connect("data/instamart.db")))==550
assert len(pd.read_sql_query(queries["Q7"],sqlite3.connect("data/instamart.db")))==183
print("ALL CHECKPOINTS PASSED")
