"""
router/scripts/generate_dataset.py
Generates a labelled dataset of queries for router training.
Labels: 0=LITE, 1=MID, 2=FULL

Run:
    python router/scripts/generate_dataset.py --out router/data/dataset.jsonl --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

# ── LITE queries (simple, short, factual, arithmetic) ─────────────────────────
LITE_TEMPLATES = [
    "What is {a} + {b}?",
    "What is {a} - {b}?",
    "What is {a} * {b}?",
    "What is {a} / {b}?",
    "What is {a} % {b}?",
    "Convert {a} kilometers to miles.",
    "Convert {a} Celsius to Fahrenheit.",
    "Convert {a} kg to pounds.",
    "Convert {a} inches to centimeters.",
    "What is the capital of {country}?",
    "How many days are in {month}?",
    "What color is {thing}?",
    "How many hours are in {a} days?",
    "What is {a} squared?",
    "What is the square root of {perfect_square}?",
    "Set a timer for {a} minutes.",
    "Remind me at {time} to {task}.",
    "Set an alarm for {time}.",
    "What time is it in {city}?",
    "Who is the president of {country}?",
    "What language is spoken in {country}?",
    "What is the currency of {country}?",
    "How far is {city} from {city2}?",
    "What year was {event} founded?",
    "Spell {word}.",
    "What does {abbr} stand for?",
    "How do you say {word} in {language}?",
    "What is the boiling point of water?",
    "What is the freezing point of water?",
    "How many sides does a {shape} have?",
    "Is {a} a prime number?",
    "What is {a} to the power of {b}?",
    "What is {fraction} as a decimal?",
    "Round {decimal} to the nearest integer.",
    "What is the absolute value of {neg}?",
    "What is the opposite of {word}?",
    "What is a synonym for {word}?",
    "Define {simple_word}.",
    "How do you spell {word}?",
    "What is 100 / {a}?",
]

LITE_FILLS = {
    "a": list(range(2, 50)),
    "b": list(range(2, 20)),
    "country": ["France","Germany","Japan","Brazil","India","Australia","Canada","Italy","Spain","Mexico","China","Russia","Egypt","Nigeria","Argentina"],
    "month": ["January","February","March","April","May","June","July","August","September","October","November","December"],
    "thing": ["the sky","grass","a banana","coal","snow","blood","a leaf","the sun","milk","coffee"],
    "perfect_square": [4,9,16,25,36,49,64,81,100,121,144,169],
    "time": ["7 AM","8:30 AM","noon","3 PM","6:45 PM","10 PM"],
    "task": ["call mom","take medicine","check email","attend meeting","go to gym"],
    "city": ["Paris","Tokyo","Mumbai","New York","London","Sydney","Dubai","Berlin","Toronto","Singapore"],
    "city2": ["Berlin","Chicago","Beijing","Lagos","Cairo"],
    "event": ["Apple","Google","Microsoft","Amazon","Tesla","UNESCO","WHO","NASA"],
    "word": ["beautiful","necessary","accommodate","separate","receive","definitely","occurrence","embarrass"],
    "abbr": ["NASA","WHO","GDP","CPU","HTML","API","URL","PDF","USB","RAM"],
    "language": ["French","Spanish","Japanese","German","Mandarin","Arabic","Hindi","Portuguese"],
    "shape": ["triangle","square","pentagon","hexagon","octagon","heptagon"],
    "fraction": ["1/2","1/4","3/4","1/8","2/5","3/8"],
    "decimal": [3.7, 2.3, 9.6, 4.1, 7.8, 5.5],
    "neg": [-5,-12,-7,-3,-99,-42],
    "simple_word": ["cat","run","blue","fast","large","small","happy","cold","wet","dry"],
}

# ── MID queries (moderate reasoning, explanation, comparison) ─────────────────
MID_TEMPLATES = [
    "Explain how {concept} works in simple terms.",
    "What is the difference between {thing1} and {thing2}?",
    "Why is {phenomenon} important?",
    "How does {process} happen?",
    "What are the main causes of {event}?",
    "Summarise the key points of {topic} in 3 bullet points.",
    "What are the pros and cons of {technology}?",
    "How would you describe {concept} to a beginner?",
    "What is {concept} and why does it matter?",
    "Compare {thing1} and {thing2} in terms of {aspect}.",
    "Give me three examples of {concept}.",
    "What are the steps to {process}?",
    "How has {topic} changed over the last decade?",
    "What are the benefits of {activity}?",
    "Why do people use {technology}?",
    "Explain the relationship between {thing1} and {thing2}.",
    "What problems does {technology} solve?",
    "How do you {task} step by step?",
    "What is the history of {topic}?",
    "What are common misconceptions about {topic}?",
    "Describe the process of {process}.",
    "What factors affect {phenomenon}?",
    "How is {thing1} different from {thing2}?",
    "What is the purpose of {concept}?",
    "Give an overview of {topic}.",
    "What are the key features of {technology}?",
    "Why is {phenomenon} a problem?",
    "How does {technology} impact {domain}?",
    "What role does {concept} play in {domain}?",
    "Explain {concept} using an analogy.",
]

MID_FILLS = {
    "concept": ["machine learning","photosynthesis","democracy","inflation","recursion","blockchain","the internet","gravity","evolution","supply and demand","neural networks","encryption","cloud computing","DNS","TCP/IP","REST APIs","version control","containerisation","microservices","agile methodology"],
    "thing1": ["Python","Java","SQL","NoSQL","HTTP","HTTPS","RAM","ROM","TCP","UDP","supervised learning","unsupervised learning","front-end","back-end","compiler","interpreter"],
    "thing2": ["JavaScript","C++","MongoDB","PostgreSQL","REST","GraphQL","CPU","GPU","IPv4","IPv6","reinforcement learning","deep learning","full-stack","DevOps","runtime","assembler"],
    "phenomenon": ["climate change","inflation","unemployment","deforestation","urbanisation","globalisation","automation","data privacy","misinformation","antibiotic resistance"],
    "process": ["digestion","blood circulation","data compression","garbage collection","model training","software deployment","database indexing","HTTP request handling","image recognition","natural language processing"],
    "event": ["the French Revolution","World War II","the 2008 financial crisis","the dot-com bubble","the Industrial Revolution","the Cold War","the COVID-19 pandemic","the space race"],
    "topic": ["renewable energy","artificial intelligence","social media","quantum computing","cybersecurity","mental health","remote work","electric vehicles","cryptocurrency","the gig economy"],
    "technology": ["Docker","Kubernetes","Git","React","TensorFlow","PostgreSQL","Redis","Kafka","Spark","GraphQL","WebSockets","OAuth","JWT","HTTPS","load balancing"],
    "aspect": ["performance","cost","scalability","ease of use","security","reliability","speed","memory usage"],
    "activity": ["exercise","meditation","reading","journalling","pair programming","code review","unit testing","continuous integration"],
    "task": ["set up a Python virtual environment","deploy a Docker container","write a unit test","optimise a SQL query","debug a memory leak","configure HTTPS","set up CI/CD","create a REST endpoint","profile Python code","migrate a database"],
    "domain": ["healthcare","finance","education","transport","manufacturing","agriculture","retail","entertainment","government","research"],
}

# ── FULL queries (multi-step, complex reasoning, long-form, ambiguous) ────────
FULL_TEMPLATES = [
    "A train leaves {city} at {time1} travelling at {speed1} km/h. Another train leaves {city2} at {time2} travelling toward it at {speed2} km/h. The distance is {dist} km. When and where do they meet?",
    "Solve: {a}x² + {b}x + {c} = 0. Show all working.",
    "A tank is {pct}% full and holds {cap} litres when full. Water drains at {rate} litres per minute. How long until empty?",
    "If {a} workers can build a wall in {b} days, how many days do {c} workers take?",
    "A shopkeeper buys goods for ₹{cost} and sells at {markup}% markup. After a {discount}% discount, what is the profit or loss?",
    "Compound interest: principal ₹{p}, rate {r}% per annum, compounded {freq}, for {n} years. Find the amount.",
    "Critically analyse the trade-offs between {thing1} and {thing2} in the context of {domain}. Include at least three dimensions.",
    "Write a detailed explanation of how {complex_concept} works, covering the underlying theory, practical applications, and limitations.",
    "Compare the architectural differences between {arch1} and {arch2} and explain when you would choose each for a production system.",
    "Design a system that {system_goal}. Describe the components, data flow, and failure modes.",
    "Explain the philosophical implications of {philosophy_topic} and how different schools of thought approach it.",
    "What are the long-term economic consequences of {policy}? Consider second-order effects.",
    "Summarise and critically evaluate the arguments for and against {debate_topic}.",
    "A ball is thrown upward at {v} m/s from a height of {h} metres. Find max height, time to reach it, and time to hit ground.",
    "Three pipes fill a tank: A in {a} hours, B in {b} hours, C empties in {c} hours. Time to fill the tank if all open together?",
    "A boat travels {d1} km upstream in {t1} hours and {d2} km downstream in {t2} hours. Find speed of boat and current.",
    "Mixture: {v1} litres of {pct1}% alcohol solution mixed with {v2} litres of {pct2}% solution. Final concentration?",
    "Explain why {complex_concept} is considered a foundational idea in {domain} and trace its historical development.",
    "What would happen if {counterfactual}? Reason through the likely chain of consequences.",
    "Analyse the causes and effects of {historical_event} from multiple perspectives.",
    "How does {complex_concept} relate to {concept2}? Draw connections and highlight contradictions.",
    "Derive the formula for {formula_topic} from first principles.",
    "A rectangle's perimeter is {perim} cm and its area is {area} cm². Find its dimensions.",
    "If the probability of event A is {pa} and event B is {pb}, and they are independent, find P(A and B), P(A or B), P(not A and B).",
    "Evaluate the integral of {integrand} with respect to x from {lower} to {upper}.",
]

FULL_FILLS = {
    "city": ["Mumbai","Delhi","Chennai","Kolkata","Hyderabad","Bangalore","Pune","Ahmedabad"],
    "city2": ["Pune","Jaipur","Lucknow","Nagpur","Surat","Bhopal","Indore","Coimbatore"],
    "time1": ["6:00 AM","7:30 AM","8:00 AM","9:15 AM","10:00 AM"],
    "time2": ["7:00 AM","8:00 AM","9:00 AM","10:30 AM","11:00 AM"],
    "speed1": [60,70,80,90,100,110,120],
    "speed2": [80,90,100,110,120,130],
    "dist": [150,180,200,240,300,350,400],
    "a": [1,2,3,1,-1,2,-3],
    "b": [-5,3,-7,4,6,-2,8],
    "c": [6,-4,10,-3,5,-8,2],
    "pct": [25,40,50,60,75,80,90],
    "cap": [500,1000,2000,5000],
    "rate": [5,10,20,25,50],
    "workers_a": [5,6,8,10,12],
    "b_days": [10,12,15,20,24],
    "c_workers": [3,4,6,8,15],
    "cost": [100,200,500,1000,2500],
    "markup": [10,15,20,25,30,40,50],
    "discount": [5,10,15,20,25],
    "p": [1000,5000,10000,25000,50000],
    "r": [5,8,10,12,15],
    "freq": ["annually","semi-annually","quarterly","monthly"],
    "n": [2,3,5,10],
    "complex_concept": ["transformers in NLP","gradient descent","the CAP theorem","Byzantine fault tolerance","the Raft consensus algorithm","quantum entanglement","CRISPR gene editing","the efficient market hypothesis","Gödel's incompleteness theorems","Bayes' theorem"],
    "arch1": ["microservices","SQL databases","actor model","event sourcing","monolithic architecture","RISC","CNN","RNN"],
    "arch2": ["monolithic architecture","NoSQL databases","shared memory","CQRS","serverless","CISC","transformer","LSTM"],
    "system_goal": ["handles 1 million concurrent users","ingests real-time sensor data from 10,000 IoT devices","serves personalised recommendations at low latency","processes financial transactions with ACID guarantees","detects anomalies in network traffic in real time"],
    "philosophy_topic": ["free will","consciousness","the nature of time","the trolley problem","personal identity","moral relativism","the simulation hypothesis","epistemic uncertainty"],
    "policy": ["universal basic income","carbon tax","open borders immigration","free trade agreements","universal healthcare","cryptocurrency regulation","algorithmic content moderation"],
    "debate_topic": ["AI replacing human jobs","nuclear energy as a climate solution","social media regulation","genetic engineering of humans","mandatory vaccination","space colonisation","surveillance capitalism"],
    "v": [10,15,20,25,30],
    "h": [0,5,10,20],
    "d1": [30,36,40,48],
    "t1": [3,4,5,6],
    "d2": [30,36,40,48],
    "t2": [2,3,4],
    "v1": [10,20,30,40],
    "pct1": [20,30,40,50,60],
    "v2": [10,20,30,40],
    "pct2": [60,70,80,90],
    "historical_event": ["the partition of India","the fall of the Roman Empire","the French Revolution","the Cold War","the 2008 financial crisis","the Green Revolution","the Arab Spring"],
    "concept2": ["information theory","category theory","game theory","thermodynamics","complexity theory","evolutionary biology","behavioural economics"],
    "formula_topic": ["the quadratic formula","compound interest","the area of a circle","Bayes' theorem","the normal distribution","Newton's second law","the time value of money"],
    "integrand": ["x²","sin(x)","e^x","1/x","x³ + 2x","cos(x)","√x"],
    "lower": [0,1,-1,0,0],
    "upper": [1,2,3,4,10],
    "perim": [28,36,40,48,56],
    "area": [48,80,96,120,192],
    "pa": [0.3,0.4,0.5,0.6,0.7],
    "pb": [0.2,0.3,0.4,0.5],
    "counterfactual": ["the internet had never been invented","antibiotics had never been discovered","electricity was a public utility like water","AI surpassed human intelligence in 2010","global temperatures rose by 4°C tomorrow"],
}


def _fill(template: str, fills: dict, rng: random.Random) -> str:
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    subs: dict[str, str] = {}
    for k in keys:
        if k in fills:
            subs[k] = str(rng.choice(fills[k]))
    result = template
    for k, v in subs.items():
        result = result.replace("{" + k + "}", v, 1)
    return result


def generate(n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    templates_fills = [
        (LITE_TEMPLATES,  LITE_FILLS,  0),
        (MID_TEMPLATES,   MID_FILLS,   1),
        (FULL_TEMPLATES,  FULL_FILLS,  2),
    ]

    for templates, fills, label in templates_fills:
        seen: set[str] = set()
        attempts = 0
        while len(seen) < n_per_class and attempts < n_per_class * 20:
            t = rng.choice(templates)
            q = _fill(t, fills, rng)
            attempts += 1
            if q not in seen:
                seen.add(q)
                rows.append({"text": q, "label": label})

    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate router training dataset")
    ap.add_argument("--out",        default="router/data/dataset.jsonl")
    ap.add_argument("--per-class",  type=int, default=220,
                    help="queries per class (default 220 → 660 total)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = generate(args.per_class, args.seed)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {0: 0, 1: 0, 2: 0}
    for r in rows:
        counts[r["label"]] += 1
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"  LITE={counts[0]}  MID={counts[1]}  FULL={counts[2]}")


if __name__ == "__main__":
    main()
