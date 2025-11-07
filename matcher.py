from dotenv import load_dotenv
import os
import numpy as np
import csv
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from dataclasses import dataclass, asdict
from groq import Groq
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# =================== Elasticsearch Client Setup ===================
# es = Elasticsearch("http://localhost:9200")
# es = Elasticsearch("https://my-elasticsearch-project-e87451.es.us-central1.gcp.elastic.cloud:443")


ES_CLOUD_URL = os.getenv("ES_CLOUD_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
GROQ_ENABLED = os.getenv("GROQ_ENABLED", "true").lower() in ("1", "true", "yes", "y")



es = Elasticsearch(ES_CLOUD_URL, api_key=ES_API_KEY, verify_certs=True)


# es = Elasticsearch(
#     os.getenv("ES_CLOUD_URL"),
#     api_key=os.getenv("ES_API_KEY"),
#     verify_certs=True
# )

# try:
#     info = es.info()
#     print("✅ Connected to Elastic Cloud:", info["cluster_name"])
# except Exception as e:
#     print("❌ Connection failed:", e)

# INDEX_NAME = "investors"
# INDEX_NAME = os.getenv("ES_INDEX_NAME", "investors")

# =================== Index Names ===================
INVESTOR_INDEX = os.getenv("ES_INVESTOR_INDEX", "investors")
STARTUP_INDEX = os.getenv("ES_STARTUP_INDEX", "startups")



# =================== Sentence Transformer Model ===================
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()

MAX_RULE_SCORE = 140
JUSTIFICATION_SCORE_THRESHOLD = 50

# ======================= Data Classes ==============================
@dataclass
class Startup:
    id: int
    user_id: int
    name: str
    industry: str
    stage: str
    location: str
    funding_goal: float
    traction: str
    team_size: int
    revenue_stage: str
    incorporation_certificate: str
    preferred_min_investor_experience: int
    preferred_min_investor_portfolio_size: int
    description: str

@dataclass
class Investor:
    id: int
    user_id: int
    name: str
    investor_type: str
    preferred_industries: str
    stage_focus: str
    min_check: float
    max_check: float
    location: str
    portfolio_size: int
    accredited_certificate: str
    total_experience: int
    preferred_revenue_stages: str
    preferred_min_team_size: int
    preferred_max_team_size: int
    description: str

# =================== CSV Data Loaders ==============================
def load_startups_from_csv(filepath="startups.csv"):
    startups = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            startups.append(Startup(
                id=int(row['id']),
                user_id=int(row['user_id']),
                name=row['name'],
                industry=row['industry'],
                stage=row['stage'],
                location=row['location'],
                funding_goal=float(row['funding_goal']),
                traction=row['traction'],
                team_size=int(row['team_size']),
                revenue_stage=row['revenue_stage'],
                incorporation_certificate=row['incorporation_certificate'],
                preferred_min_investor_experience=int(row['preferred_min_investor_experience']),
                preferred_min_investor_portfolio_size=int(row['preferred_min_investor_portfolio_size']),
                description=row['description']
            ))
    return startups

def load_investors_from_csv(filepath="investors.csv"):
    investors = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            investors.append(Investor(
                id=int(row['id']),
                user_id=int(row['user_id']),
                name=row['name'],
                investor_type=row['investor_type'],
                preferred_industries=row['preferred_industries'],
                stage_focus=row['stage_focus'],
                min_check=float(row['min_check']),
                max_check=float(row['max_check']),
                location=row['location'],
                portfolio_size=int(row['portfolio_size']),
                accredited_certificate=row['accredited_certificate'],
                total_experience=int(row['total_experience']),
                preferred_revenue_stages=row['preferred_revenue_stages'],
                preferred_min_team_size=int(row['preferred_min_team_size']),
                preferred_max_team_size=int(row['preferred_max_team_size']),
                description=row['description']
            ))
    return investors

# ==================== Embedding Computation ========================
def compute_embedding(text: str) -> np.ndarray:
    emb = model.encode([text], normalize_embeddings=True)
    return np.array(emb).astype("float32")[0]

# ================== Rule-based Scoring =============================
def rule_based_score(startup: Startup, investor: Investor) -> float:
    score = 0.0
    s_industries = set(i.strip().lower() for i in startup.industry.split(","))
    i_industries = set(i.strip().lower() for i in investor.preferred_industries.split(","))
    union = s_industries.union(i_industries)
    if union:
        intersection = s_industries.intersection(i_industries)
        score += len(intersection) / len(union) * 40

    investor_stages = [s.strip().lower() for s in investor.stage_focus.split(",")]
    if startup.stage.lower() in investor_stages:
        score += 30

    if startup.location.lower() == investor.location.lower():
        score += 10

    if investor.min_check <= startup.funding_goal <= investor.max_check:
        score += 20

    if investor.preferred_min_team_size <= startup.team_size <= investor.preferred_max_team_size:
        score += 10

    preferred_revenue_stages = [r.strip().lower() for r in investor.preferred_revenue_stages.split(",")]
    if startup.revenue_stage.lower() in preferred_revenue_stages:
        score += 10

    portfolio_points = min(investor.portfolio_size / 50, 1.0) * 10
    score += portfolio_points

    experience_points = min(investor.total_experience / 30, 1.0) * 10
    score += experience_points

    normalized_score = (score / MAX_RULE_SCORE) * 100
    return min(normalized_score, 100.0)

# ================== Elasticsearch Setup ===================
def create_index(index_name: str):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text"},
                "description": {"type": "text"},
                "description_vector": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)

def index_startups(startups):
    for s in startups:
        emb = compute_embedding(s.description).tolist()
        doc = {
            "id": str(s.id),
            "name": s.name,
            "description": s.description,
            "description_vector": emb
        }
        es.index(index=STARTUP_INDEX, id=s.id, document=doc)
    es.indices.refresh(index=STARTUP_INDEX)



def index_investors(investors):
    for inv in investors:
        emb = compute_embedding(inv.description).tolist()
        doc = {
            "id": str(inv.id),
            "name": inv.name,
            "description": inv.description,
            "description_vector": emb
        }
        es.index(index=INVESTOR_INDEX, id=inv.id, document=doc)
    es.indices.refresh(index=INVESTOR_INDEX)

# =================== Elasticsearch Vector Search ====================
# def search_similar_investors(startup_emb, top_k=5):
#     query = {
#         "size": top_k,
#         "query": {
#             "script_score": {
#                 "query": {"match_all": {}},
#                 "script": {
#                     "source": "cosineSimilarity(params.query_vector, 'description_vector') + 1.0",
#                     "params": {"query_vector": startup_emb}
#                 }
#             }
#         }
#     }
#     result = es.search(index=INDEX_NAME, body=query)
#     hits = result["hits"]["hits"]
#     return hits


def search_similar_investors(startup_emb, top_k=5):
    query = {
        "knn": {
            "field": "description_vector",
            "query_vector": startup_emb,
            "k": top_k,
            "num_candidates": 50
        }
    }
    result = es.search(index=INVESTOR_INDEX, body=query)
    return result["hits"]["hits"]

def search_similar_startups(investor_emb, top_k=5):
    query = {
        "knn": {
            "field": "description_vector",
            "query_vector": investor_emb,
            "k": top_k,
            "num_candidates": 50
        }
    }
    result = es.search(index=STARTUP_INDEX, body=query)
    return result["hits"]["hits"]



# =================== LLM Justification Integration ====================
def get_llm_justification(investor, startup, rule_score, semantic_score, final_score):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "❌ Error: GROQ_API_KEY not set."

    client = Groq(api_key=GROQ_API_KEY)

    # Prepare data for LLM
    investor_data = {
        "name": investor.name,
        "type": investor.investor_type,
        "preferred_industries": investor.preferred_industries,
        "stage_focus": investor.stage_focus,
        "min_check": investor.min_check,
        "max_check": investor.max_check,
        "location": investor.location,
        "portfolio_size": investor.portfolio_size,
        "accredited_certificate": investor.accredited_certificate,
        "total_experience": investor.total_experience,
        "preferred_revenue_stages": investor.preferred_revenue_stages,
        "preferred_team_size_range": f"{investor.preferred_min_team_size}-{investor.preferred_max_team_size}",
        "description": investor.description
    }

    startup_data = {
        "name": startup.name,
        "industry": startup.industry,
        "stage": startup.stage,
        "location": startup.location,
        "funding_goal": startup.funding_goal,
        "traction": startup.traction,
        "team_size": startup.team_size,
        "revenue_stage": startup.revenue_stage,
        "incorporation_certificate": startup.incorporation_certificate,
        "description": startup.description
    }

    prompt = f"""
Role: You are an expert financial analyst specializing in investor-startup matching. Your goal is to provide a clear, concise, and professional analysis of the compatibility between a startup and an investor profile.

Instructions:
Analyze the provided investor and startup profiles.
Write a structured report with the following sections:

Overall Match Summary: Begin with a single sentence giving a quick conclusion on the match quality (e.g., Strong Match, Good Match with Caveats, Weak Match).

Key Justifications: Provide a bulleted list of the top 3-5 reasons why this is a strong match. Focus on the most critical factors like industry, investment stage, check size, traction, and team. Use specific data from the profiles (e.g., "The investor's focus on Fintech perfectly aligns with the startup's industry.").

Considerations & Warnings: Provide a bulleted list of any potential warnings or considerations. Gently note if the investor is not accredited or if the startup is not incorporated. Also note other soft negatives like geographic distance, but frame them as "considerations" rather than outright rejections (e.g., "Note: The geographic distance between New York and Bangalore may require additional effort for relationship management.").

Final Verdict: End with a 1-2 sentence conclusion that summarizes the analysis and gives a clear, actionable recommendation (e.g., "We highly recommend proceeding.").

Tone: Professional, analytical, and objective. Warnings should be clear but gentle and suggestive, not alarmist.

Data Format:
The input will be provided in the following JSON format. Use the data within to support your analysis.
{{
"investor": {json.dumps(investor_data, indent=2)},
"startup": {json.dumps(startup_data, indent=2)}
}}

Fit Scores: Rule={rule_score:.2f}, Semantic={semantic_score:.2f}, Final={final_score:.2f}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Updated model name
            messages=messages,
            temperature=0.2,
            max_tokens=500,  # Increased token limit for detailed response
            top_p=0.95,
            stream=False
        )

        justification = completion.choices[0].message.content.strip()
        return justification or "Justification not available."

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

# =================== Hybrid Matching Engine (Startup to investor) =========================
def match_startups_to_investors(top_k=5):
    matches = []
    startups = load_startups_from_csv()
    investors = load_investors_from_csv()

    # (re)create and populate the investors index
    create_index(INVESTOR_INDEX)          # <- uses your refactored create_index(index_name)
    index_investors(investors)

    print("🔍 Matching Startups → Investors ...")
    for startup in startups:
        startup_emb = compute_embedding(startup.description).tolist()

        # optional prefilter via rule score > 0 (keeps your old behavior)
        viable_ids = {inv.id for inv in investors if rule_based_score(startup, inv) > 0}
        if not viable_ids:
            continue

        es_hits = search_similar_investors(startup_emb, top_k)
        filtered_hits = [h for h in es_hits if int(h["_id"]) in viable_ids]

        for hit in filtered_hits:
            inv_id = int(hit["_id"])
            investor = next((inv for inv in investors if inv.id == inv_id), None)
            if investor is None:
                continue

            # same scaling you used previously
            # semantic_score = (hit["_score"] - 1.0) * 50
            semantic_score = ((hit["_score"] + 1) / 2) * 100  # normalize cosine similarity (-1 to 1 → 0 to 100)
            rule_score = rule_based_score(startup, investor)
            final_score = 0.7 * rule_score + 0.3 * semantic_score

            if GROQ_ENABLED and final_score >= JUSTIFICATION_SCORE_THRESHOLD:
                justification = get_llm_justification(investor, startup, rule_score, semantic_score, final_score)
            else:
                justification = "Match score below threshold; no justification generated."

            matches.append({
                "direction": "startup→investor",
                "startup_id": startup.id,
                "startup_name": startup.name,
                "investor_id": investor.id,
                "investor_name": investor.name,
                "rule_score": round(rule_score, 2),
                "semantic_score": round(semantic_score, 2),
                "final_score": round(final_score, 2),
                "justification": justification
            })

    matches.sort(key=lambda x: x["final_score"], reverse=True)
    return matches


# =================== Hybrid Matching Engine (Investor to Startup) =========================
def match_investors_to_startups(top_k=5):
    matches = []
    startups = load_startups_from_csv()
    investors = load_investors_from_csv()

    # (re)create and populate the startups index
    create_index(STARTUP_INDEX)
    index_startups(startups)

    print("🔍 Matching Investors → Startups ...")
    for investor in investors:
        investor_emb = compute_embedding(investor.description).tolist()

        es_hits = search_similar_startups(investor_emb, top_k)
        for hit in es_hits:
            s_id = int(hit["_id"])
            startup = next((s for s in startups if s.id == s_id), None)
            if startup is None:
                continue

            semantic_score = (hit["_score"] - 1.0) * 50
            rule_score = rule_based_score(startup, investor)
            final_score = 0.7 * rule_score + 0.3 * semantic_score

            if GROQ_ENABLED and final_score >= JUSTIFICATION_SCORE_THRESHOLD:
                justification = get_llm_justification(investor, startup, rule_score, semantic_score, final_score)
            else:
                justification = "Match score below threshold; no justification generated."

            matches.append({
                "direction": "investor→startup",
                "startup_id": startup.id,
                "startup_name": startup.name,
                "investor_id": investor.id,
                "investor_name": investor.name,
                "rule_score": round(rule_score, 2),
                "semantic_score": round(semantic_score, 2),
                "final_score": round(final_score, 2),
                "justification": justification
            })

    matches.sort(key=lambda x: x["final_score"], reverse=True)
    return matches


# =================== Hybrid Matching Engine (Both Directions) =========================
def two_way_match(top_k=5):
    m1 = match_startups_to_investors(top_k)
    m2 = match_investors_to_startups(top_k)
    all_matches = m1 + m2
    all_matches.sort(key=lambda x: x["final_score"], reverse=True)
    print(f"✅ Total matches (both directions): {len(all_matches)}")
    return all_matches

# ===================== Display Top 50 Matches =======================
def top_50_best_matches(matches: List[Dict[str, Any]], save_json: bool = True):
    top_matches = matches[:50]

    print(f"\n{'Startup':<20} {'Investor':<25} {'Rule':<8} {'Semantic':<10} {'Final':<8} {'Justification'}")
    print("-" * 120)

    for m in top_matches:
        if m['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            just_display = "✅ Justification available"
        else:
            just_display = "❌ Below threshold"
        print(f"{m['startup_name']:<20} {m['investor_name']:<25} {m['rule_score']:<8.2f} {m['semantic_score']:<10.2f} {m['final_score']:<8.2f} {just_display}")

    print(f"\n✅ Total top matches displayed: {len(top_matches)}")
    
    # Save all matches to JSON file
    if save_json:
        filename = "matches_output.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        print(f"💾 All matches saved to {filename}")
    
    return top_matches

# ===================== Startup Match Lookup =======================
def startup_match_lookup(matches: List[Dict[str, Any]], startups: List[Startup]):
    startup_map = {s.id: s for s in startups}
    startup_ids = sorted(set(m['startup_id'] for m in matches))

    print("\nSelect a startup to see matched investors:\n")
    # Show only first 50 startups
    displayed_startups = startup_ids[:50]
    for i, s_id in enumerate(displayed_startups, 1):
        print(f"{i:2d}. {startup_map[s_id].name}")

    choice = input("\nEnter choice number (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        return False

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(displayed_startups):
            print("Invalid choice.")
            return True
    except ValueError:
        print("Invalid input.")
        return True

    s_id = displayed_startups[idx]
    startup = startup_map[s_id]

    print(f"\nInvestors matched to startup: {startup.name}\n")
    print(f"{'Investor':<25} {'Rule':<8} {'Semantic':<10} {'Final':<8} {'Justification'}")
    print("-" * 100)

    filtered = [m for m in matches if m['startup_id'] == s_id]
    for m in filtered:
        if m['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            just_display = "✅ Justification available"
        else:
            just_display = "❌ Below threshold"
        print(f"{m['investor_name']:<25} {m['rule_score']:<8.2f} {m['semantic_score']:<10.2f} {m['final_score']:<8.2f} {just_display}")

    # Show detailed justifications
    print(f"\n📝 DETAILED JUSTIFICATIONS FOR {startup.name}:\n")
    print("="*100)
    for i, m in enumerate(filtered, 1):
        if m['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            print(f"\n{i}. Investor: {m['investor_name']}")
            print(f"   Final Score: {m['final_score']:.2f}/100")
            print(f"   Justification:\n      {m['justification']}\n")
            print("-" * 100)

    print(f"\nTotal investors matched: {len(filtered)}")
    return True

# ===================== Investor Match Lookup =======================
def investor_match_lookup(matches: List[Dict[str, Any]], investors: List[Investor]):
    investor_map = {inv.id: inv for inv in investors}
    investor_ids = sorted(set(m['investor_id'] for m in matches))

    print("\nSelect an investor to see matched startups:\n")
    # Show only first 50 investors
    displayed_investors = investor_ids[:50]
    for i, inv_id in enumerate(displayed_investors, 1):
        print(f"{i:2d}. {investor_map[inv_id].name}")

    choice = input("\nEnter choice number (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        return False

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(displayed_investors):
            print("Invalid choice.")
            return True
    except ValueError:
        print("Invalid input.")
        return True

    inv_id = displayed_investors[idx]
    investor = investor_map[inv_id]

    print(f"\nStartups matched to investor: {investor.name}\n")
    print(f"{'Startup':<25} {'Rule':<8} {'Semantic':<10} {'Final':<8} {'Justification'}")
    print("-" * 100)

    filtered = [m for m in matches if m['investor_id'] == inv_id]
    for m in filtered:
        if m['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            just_display = "✅ Justification available"
        else:
            just_display = "❌ Below threshold"
        print(f"{m['startup_name']:<25} {m['rule_score']:<8.2f} {m['semantic_score']:<10.2f} {m['final_score']:<8.2f} {just_display}")

    # Show detailed justifications
    print(f"\n📝 DETAILED JUSTIFICATIONS FOR {investor.name}:\n")
    print("="*100)
    for i, m in enumerate(filtered, 1):
        if m['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            print(f"\n{i}. Startup: {m['startup_name']}")
            print(f"   Final Score: {m['final_score']:.2f}/100")
            print(f"   Justification:\n      {m['justification']}\n")
            print("-" * 100)

    print(f"\nTotal startups matched: {len(filtered)}")
    return True

# ===================== Specific Match Lookup =======================
def specific_match_lookup(matches: List[Dict[str, Any]], startups: List[Startup], investors: List[Investor]):
    startup_map = {s.id: s for s in startups}
    investor_map = {inv.id: inv for inv in investors}
    
    # Get startup name
    startup_name = input("\nEnter startup name (or 'q' to quit): ").strip()
    if startup_name.lower() == 'q':
        return False
    
    # Find matching startups (partial match)
    matching_startups = [s for s in startups if startup_name.lower() in s.name.lower()]
    if not matching_startups:
        print("No startups found with that name.")
        return True
    
    if len(matching_startups) > 1:
        print("\nMultiple startups found:")
        for i, s in enumerate(matching_startups, 1):
            print(f"{i}. {s.name}")
        try:
            choice = int(input("Select startup number: ")) - 1
            if choice < 0 or choice >= len(matching_startups):
                print("Invalid choice.")
                return True
            startup = matching_startups[choice]
        except ValueError:
            print("Invalid input.")
            return True
    else:
        startup = matching_startups[0]
    
    # Get investor name
    investor_name = input("Enter investor name: ").strip()
    if investor_name.lower() == 'q':
        return False
    
    # Find matching investors (partial match)
    matching_investors = [inv for inv in investors if investor_name.lower() in inv.name.lower()]
    if not matching_investors:
        print("No investors found with that name.")
        return True
    
    if len(matching_investors) > 1:
        print("\nMultiple investors found:")
        for i, inv in enumerate(matching_investors, 1):
            print(f"{i}. {inv.name}")
        try:
            choice = int(input("Select investor number: ")) - 1
            if choice < 0 or choice >= len(matching_investors):
                print("Invalid choice.")
                return True
            investor = matching_investors[choice]
        except ValueError:
            print("Invalid input.")
            return True
    else:
        investor = matching_investors[0]
    
    # Find match
    match = next((m for m in matches 
                  if m['startup_id'] == startup.id and m['investor_id'] == investor.id), 
                 None)
    
    if match:
        print(f"\n.MATCH FOUND BETWEEN {startup.name} AND {investor.name}:\n")
        print(f"Rule Score: {match['rule_score']:.2f}")
        print(f"Semantic Score: {match['semantic_score']:.2f}")
        print(f"Final Score: {match['final_score']:.2f}")
        if match['final_score'] >= JUSTIFICATION_SCORE_THRESHOLD:
            print(f"\nJustification:\n{match['justification']}")
        else:
            print("Justification: Below threshold")
    else:
        print(f"\nNo match found between {startup.name} and {investor.name}")
    
    return True

# ========================= Command Line Interface ==============================
# if __name__ == "__main__":
#     print("🚀 Starting Hybrid Matching Engine...")
#     matches = hybrid_match(top_k=30)
if __name__ == "__main__":
    print("🚀 Starting Two-Way Hybrid Matching Engine...")
    matches = two_way_match(top_k=30)
    _ = top_50_best_matches(matches, save_json=True)
    # ... keep the CLI below if you want
    
    # Display top 50 matches and save all to JSON
    # _ = top_50_best_matches(matches, save_json=True)
    
    startups = load_startups_from_csv()
    investors = load_investors_from_csv()

    while True:
        print("\n" + "="*60)
        print("             MATCH LOOKUP MENU")
        print("="*60)
        print("1. Lookup by Investor")
        print("2. Lookup by Startup")
        print("3. Specific Startup-Investor Match")
        print("4. Show Top Matches Again")
        print("5. Exit")
        print("-"*60)

        choice = input("➡️ Select option (1-5): ").strip()

        if choice == "1":
            while investor_match_lookup(matches, investors):
                pass
        elif choice == "2":
            while startup_match_lookup(matches, startups):
                pass
        elif choice == "3":
            while specific_match_lookup(matches, startups, investors):
                pass
        elif choice == "4":
            _ = top_50_best_matches(matches, save_json=False)
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please choose 1-5.")