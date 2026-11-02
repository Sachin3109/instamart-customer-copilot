import os,re,json,random,time
import pandas as pd
import mock_llm
from mock_llm import est_tokens,TransientLLMError
os.makedirs("outputs",exist_ok=True)
T=pd.read_csv("data/raw/tickets.csv",keep_default_na=False)
segments=pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
patterns=[(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),"[EMAIL]"),(re.compile(r"\b[6-9]\d{9}\b"),"[PHONE]"),(re.compile(r"ORD-\d{6}"),"[ORDER_ID]")]
def redact(s):
    counts=[]
    for pat,repl in patterns: s,n=pat.subn(repl,s); counts.append(n)
    return s,counts
zero="You are a CLASSIFY assistant. Choose exactly one category: billing, delivery, technical, fraud. Reply JSON only: {\"category\": \"...\", \"reason\": \"...\"}."
few=zero+"\nEXAMPLES:\nFraud example: unauthorized card transaction -> {\"category\":\"fraud\",\"reason\":\"unauthorized transaction\"}\nDelivery example: rider is late -> {\"category\":\"delivery\",\"reason\":\"late delivery\"}"
def run(prompt,budget=None):
    mock_llm.reset(); total=0; out=[]
    for _,r in T.iterrows():
        text=str(r.text)
        if not text.strip(): out.append(dict(ticket_id=r.ticket_id,customer_id=r.customer_id,status="skipped_empty",category="",tokens_used=0)); continue
        if est_tokens(text)>400: out.append(dict(ticket_id=r.ticket_id,customer_id=r.customer_id,status="flagged_long",category="",tokens_used=0)); continue
        red,counts=redact(text)
        assert not any(p.search(red) for p,_ in patterns)
        assert red==redact(text)[0]
        if budget is not None and total>=budget: out.append(dict(ticket_id=r.ticket_id,customer_id=r.customer_id,status="budget_exhausted",category="",tokens_used=0)); continue
        msgs=[{"role":"system","content":prompt},{"role":"user","content":red}]
        tries=0
        while True:
            try: response=mock_llm.chat(msgs); break
            except TransientLLMError:
                if tries>=3: raise
                wait=.05*(2**tries)*random.Random(42+tries).uniform(.9,1.15); print(f"retry {r.ticket_id} retry={tries} wait={wait:.4f}s"); time.sleep(wait); tries+=1
        used=response.usage.total_tokens; total+=used
        raw=response.choices[0].message.content or ""; cat=""; status="flagged_invalid_output"
        try:
            obj=json.loads(raw); candidate=obj.get("category","")
            if candidate in ["billing","delivery","technical","fraud"]: cat=candidate; status="classified_json"
        except Exception:
            words=re.findall(r"[a-z]+",raw.lower()); hits=[x for x in ["billing","delivery","technical","fraud"] if x in words]
            if len(hits)==1: cat=hits[0]; status="classified_keyword"
        out.append(dict(ticket_id=r.ticket_id,customer_id=r.customer_id,status=status,category=cat,tokens_used=used))
    return pd.DataFrame(out),total
z,zt=run(zero); f,ft=run(few)
print("zero",z.status.value_counts().to_dict(),"accuracy",((z.merge(T[["ticket_id","true_category"]],on="ticket_id").query("status in ['classified_json','classified_keyword']").eval("category==true_category")).mean()),"tokens",zt)
print("few",f.status.value_counts().to_dict(),"accuracy",((f.merge(T[["ticket_id","true_category"]],on="ticket_id").query("status in ['classified_json','classified_keyword']").eval("category==true_category")).mean()),"tokens",ft)
budget=int(os.getenv("TOKEN_BUDGET","5000")); b,_=run(zero,budget); b.to_csv("outputs/triage_budget_demo.csv",index=False); print("budget exhausted",sum(b.status=="budget_exhausted"),"budget",budget)
# Final few-shot with ample budget, segment-based routing
f,_=run(few,100000)
rows=[]
for _,r in f.iterrows():
    seg=segments.loc[r.customer_id] if r.customer_id in segments.index else None
    sname="" if seg is None else seg.segment_name
    premium=False if seg is None else bool(seg.is_premium)
    if r.status in ["skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"]: route,priority="human_review","NORMAL"
    elif r.category=="fraud": route,priority="security_team","HIGH"
    elif premium: route,priority="premium_desk","HIGH"
    else: route,priority="support_queue","NORMAL"
    rows.append([r.ticket_id,r.customer_id,r.status,r.category,sname,priority,route,r.tokens_used])
pd.DataFrame(rows,columns=["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]).to_csv("outputs/triage_results.csv",index=False)
