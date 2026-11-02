import os,re,json,time,sqlite3,uuid
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
os.makedirs("outputs",exist_ok=True); os.makedirs("state",exist_ok=True)
def make_chunks(overlap=10):
    chunks=[]
    for fn in sorted(os.listdir("data/policies")):
        words=open("data/policies/"+fn,encoding="utf-8").read().split()
        start=0
        while start<len(words):
            part=words[start:start+40]; chunks.append({"source":fn,"text":" ".join(part)})
            if start+40>=len(words): break
            start+=40-overlap
    return chunks
chunks=make_chunks(); vectorizer=TfidfVectorizer().fit([x["text"] for x in chunks]); matrix=vectorizer.transform([x["text"] for x in chunks])
def retrieve(query,k=3,overlap_chunks=None,vec=None,mat=None):
    ch=overlap_chunks or chunks; v=vec or vectorizer; m=mat if mat is not None else matrix
    q=v.transform([query]); sims=cosine_similarity(q,m).ravel(); ix=sims.argsort()[::-1][:k]
    return [{"source":ch[i]["source"],"score":float(sims[i]),"text":ch[i]["text"]} for i in ix]
def answer(question):
    hits=retrieve(question,3); context="\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    msg=[{"role":"system","content":"POLICY ASSISTANT: Answer only using supplied context. If unavailable, say you could not find the information."},{"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    return mock_llm.chat(msg).choices[0].message.content,[h["source"] for h in hits]
ev=pd.read_csv("data/raw/policy_eval.csv",keep_default_na=False)
res=[]
for _,r in ev.iterrows():
    ans,src=answer(r.question)
    res.append((r.expected_doc in src if r.expected_doc!="NONE" else None, r.expected_fact in ans if r.expected_fact else ans.startswith("I could not find")))
print("RAG chunks",len(chunks),"eval retrieval",sum(x[0] is True for x in res[:12]),"/12","facts",sum(x[1] for x in res[:12]),"/12","fallback",sum(x[1] for x in res[12:]),"/2")
# tools and map
def get_order(order_id):
    con=sqlite3.connect("data/instamart.db"); row=con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone(); con.close()
    return {"error":"order not found"} if row is None else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
def search_policy(query): return {"results":retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes): return {"refund_inr":round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
def ledger(tool,order_id,refund_inr):
    with open("outputs/refund_ledger.jsonl","a") as f: f.write(json.dumps({"timestamp":time.time(),"tool":tool,"order_id":order_id,"refund_inr":refund_inr})+"\n")
    return {"refund_inr":refund_inr}
def issue_refund(order_id,refund_inr): return ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
TOOLS_SPEC=[{"name":n,"description":f"Execute {n} for refund processing","parameters":{"type":"object","properties":{}}} for n in TOOL_MAP]
def checkpoint(run_id,scenario_id,messages,tokens,step,executed):
    with open(f"state/{run_id}.json","w") as f: json.dump({"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,"step":step,"executed":executed},f)
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None,resume=False):
    rid=run_id or scenario_id; path=f"state/{rid}.json"
    if resume:
        st=json.load(open(path)); messages=st["messages"]; tokens=st["tokens_used"]; step=st["step"]; executed=st["executed"]
    else:
        messages=[{"role":"system","content":"AGENT: Use tools to verify and process refund requests safely."},{"role":"user","content":user_message}]; tokens=0; step=0; executed=[]
    start=time.monotonic(); observations=[]
    for _ in range(max_iterations-step):
        if time.monotonic()-start>=time_limit_seconds: return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to a human agent.","steps":observations,"llm_calls":step,"tokens_used":tokens}
        response=mock_llm.chat(messages); step+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit: return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: human review required.","steps":observations,"llm_calls":step,"tokens_used":tokens}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or ""; status=content.split(":")[0]
            return {"status":status,"final_answer":content,"steps":observations,"llm_calls":step,"tokens_used":tokens}
        calls=[{"id":tc.id,"name":tc.name,"arguments":tc.arguments} for tc in msg.tool_calls]
        messages.append({"role":"assistant","content":None,"tool_calls":calls})
        for tc in msg.tool_calls:
            args=tc.arguments; obs=TOOL_MAP[tc.name](**args); executed.append(tc.name); observations.append({"tool":tc.name,"arguments":args,"observation":obs})
            messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(obs)})
            checkpoint(rid,scenario_id,messages,tokens,step,executed)
            if crash_after and len(executed)==crash_after: raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to a human agent.","steps":observations,"llm_calls":step,"tokens_used":tokens}
def resume_agent(run_id):
    st=json.load(open(f"state/{run_id}.json")); user=next(x["content"] for x in st["messages"] if x["role"]=="user")
    return run_agent(user,st["scenario_id"],run_id=run_id,resume=True)
sc=pd.read_csv("data/raw/agent_scenarios.csv")
if os.path.exists("outputs/refund_ledger.jsonl"): os.remove("outputs/refund_ledger.jsonl")
runs=[]
for _,r in sc.iterrows():
    mock_llm.reset(); result=run_agent(r.message,r.scenario_id)
    runs.append({"scenario_id":r.scenario_id,**result})
with open("outputs/agent_runs.jsonl","w") as f:
    for x in runs: f.write(json.dumps(x)+"\n")
print("agent",[(x["scenario_id"],x["status"],x["llm_calls"]) for x in runs])
# safety demonstrations
mock_llm.reset(); print("token demo",run_agent(sc.iloc[0].message,"S1_token_demo",token_limit=200)["status"])
mock_llm.reset(); print("timeout demo",run_agent(sc.iloc[0].message,"S1_timeout_demo",time_limit_seconds=0)["status"])
# crash/resume: remove ledger, run first two tools then continue
mock_llm.reset()
try: run_agent(sc.iloc[0].message,"S1_resume",run_id="S1_resume",crash_after=2)
except RuntimeError: pass
mock_llm.reset(); resumed=resume_agent("S1_resume"); print("resume",resumed["status"],resumed["final_answer"])
